#!/usr/bin/env python3
"""
download_nyc_311_noise.py

Class-based, restartable, date-partitioned downloader for NYC 311 noise complaints
via Socrata (SODA v2).

Key features
- Partitions by day/week/month into separate CSV files (restartable with resume=True)
- Paginates within each partition using LIMIT/OFFSET
- Supports start_date / end_date (end exclusive): [start, end)
- Uses X-App-Token (NYC_APP_TOKEN) if available
- Uses request_with_retry(session, method, url, ...) from helpers.py
- Injects time filter into base SoQL safely before tail clauses

Example (CLI):
  python scripts/ingest/download_nyc_311_noise.py \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --freq month \
    --resume

Example (Python):
  dl = NYC311NoiseDownloader(
      query_file="./queries/nyc_311_noise.soql",
      out_dir="data/raw/nyc/311_noise",
      freq="month",
      limit=2000,
      resume=True,
  )
  total = dl.run("2023-01-01", "2024-01-01")
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from helpers import load_soql, request_with_retry

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
OUT_DIR = "./data/raw/nyc/311_noise"
# ---------------------------------------------------------------------
# SoQL helpers (module-level regex; used by class)
# ---------------------------------------------------------------------

_TAIL_CLAUSE_RE = re.compile(
    r"\b(ORDER\s+BY|SEARCH|GROUP\s+BY|HAVING|LIMIT|OFFSET)\b",
    flags=re.IGNORECASE,
)
_WHERE_RE = re.compile(r"\bWHERE\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime  # exclusive


class NYC311NoiseDownloader:
    """
    Downloader for NYC 311 noise complaints (Socrata SODA v2), partitioned by time windows.

    Responsibilities:
    - Load base SoQL (from file or provided string)
    - Create/own a requests.Session with optional X-App-Token
    - Iterate time windows
    - Download each partition with LIMIT/OFFSET pagination into CSV(s)
    """

    def __init__(
        self,
        *,
        query_file: str = "./queries/nyc_311_noise.soql",
        out_dir: str = "data/raw/nyc/311_noise",
        freq: str = "month",
        limit: int = 1000,
        resume: bool = False,
        max_retries: int = 6,
        timeout: int = 120,
        debug: bool = False,
        base_url: str = BASE_URL,
        app_token: Optional[str] = None,
        base_soql: Optional[str] = None,
    ) -> None:
        """
        Args:
            query_file: Path to SoQL file (ignored if base_soql is provided).
            out_dir: Base output directory.
            freq: Partition frequency: "day" | "week" | "month".
            limit: Pagination page size for LIMIT/OFFSET.
            resume: If True, skip partitions whose output file already exists.
            max_retries: Max retries per request in request_with_retry.
            timeout: Request timeout seconds.
            debug: Print first-page SoQL for each partition.
            base_url: SODA endpoint URL.
            app_token: Optional Socrata app token. If None, reads NYC_APP_TOKEN from env.
            base_soql: Optional SoQL string. If provided, query_file is not used.
        """
        if freq not in ("day", "week", "month"):
            raise ValueError("freq must be one of: day, week, month")

        self.query_file = query_file
        self.out_dir = Path(out_dir)
        self.freq = freq
        self.limit = int(limit)
        self.resume = bool(resume)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self.debug = bool(debug)
        self.base_url = base_url

        self._base_soql = base_soql.strip() if isinstance(base_soql, str) else None

        # Token: argument overrides env
        self.app_token = (
            app_token if app_token is not None else os.getenv("NYC_APP_TOKEN")
        )

        # Lazily created session
        self._session: Optional[requests.Session] = None

    # -------------------------
    # Public API
    # -------------------------

    def run(self, start_date: str, end_date: str) -> int:
        """
        Run downloader from start_date (inclusive) to end_date (exclusive).

        Args:
            start_date: "YYYY-MM-DD" inclusive
            end_date: "YYYY-MM-DD" exclusive boundary

        Returns:
            Total rows downloaded across all windows.
        """
        base_soql = self._load_base_soql()

        start_dt = self.parse_date_ymd(start_date)
        end_dt = self.parse_date_ymd(end_date)

        part_dir = self.out_dir / f"by_{self.freq}"
        part_dir.mkdir(parents=True, exist_ok=True)

        session = self._get_session()

        grand_total = 0
        for w in self.iter_windows(start_dt, end_dt, self.freq):
            label = self.window_label(w, self.freq)
            out_file = part_dir / f"{label}.csv"

            if self.resume and out_file.exists():
                print(f"Skipping (exists): {out_file}")
                continue

            print(
                f"Downloading window [{w.start.date()} → {w.end.date()}) to {out_file}"
            )
            n = self.download_partition(
                session=session,
                base_soql=base_soql,
                window=w,
                limit=self.limit,
                out_file=out_file,
                max_retries=self.max_retries,
                timeout=self.timeout,
                debug=self.debug,
            )
            print(f"  Completed {out_file.name}: {n} rows")
            grand_total += n

        print(f"\nDone. Total rows downloaded across windows: {grand_total}")
        return grand_total

    # -------------------------
    # Session / config
    # -------------------------

    def _get_session(self) -> requests.Session:
        if self._session is not None:
            return self._session

        s = requests.Session()
        headers = {"Accept": "application/json"}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        s.headers.update(headers)
        self._session = s
        return s

    def _load_base_soql(self) -> str:
        if self._base_soql:
            return self._base_soql
        q = load_soql(self.query_file).strip()
        if not q:
            raise ValueError(f"Empty SoQL loaded from: {self.query_file}")
        return q

    # -------------------------
    # SoQL utilities
    # -------------------------

    @staticmethod
    def inject_time_filter(base_soql: str, time_predicate: str) -> str:
        """
        Insert time_predicate into WHERE (or create WHERE) before any tail clauses
        (ORDER BY / SEARCH / GROUP BY / HAVING / LIMIT / OFFSET).
        """
        q = base_soql.strip().rstrip(";")

        m = _TAIL_CLAUSE_RE.search(q)
        if m:
            head = q[: m.start()].rstrip()
            tail = q[m.start() :].lstrip()
        else:
            head = q
            tail = ""

        if _WHERE_RE.search(head):
            return f"{head} AND {time_predicate} {tail}".strip()
        return f"{head} WHERE {time_predicate} {tail}".strip()

    @staticmethod
    def soql_time_predicate(start: datetime, end: datetime) -> str:
        s0 = start.strftime("%Y-%m-%dT%H:%M:%S")
        s1 = end.strftime("%Y-%m-%dT%H:%M:%S")
        return f"created_date >= '{s0}' AND created_date < '{s1}'"

    # -------------------------
    # Date partitioning
    # -------------------------

    @staticmethod
    def parse_date_ymd(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    @staticmethod
    def iter_windows(start: datetime, end: datetime, freq: str) -> Iterator[Window]:
        cur = start

        if freq == "day":
            step = timedelta(days=1)
            while cur < end:
                nxt = min(cur + step, end)
                yield Window(cur, nxt)
                cur = nxt
            return

        if freq == "week":
            step = timedelta(days=7)
            while cur < end:
                nxt = min(cur + step, end)
                yield Window(cur, nxt)
                cur = nxt
            return

        if freq == "month":
            while cur < end:
                y, m = cur.year, cur.month
                if m == 12:
                    nxt = datetime(y + 1, 1, 1)
                else:
                    nxt = datetime(y, m + 1, 1)
                nxt = min(nxt, end)
                yield Window(cur, nxt)
                cur = nxt
            return

        raise ValueError("freq must be one of: day, week, month")

    @staticmethod
    def window_label(w: Window, freq: str) -> str:
        if freq in ("day", "week"):
            return w.start.strftime("%Y-%m-%d")
        if freq == "month":
            return w.start.strftime("%Y-%m")
        raise ValueError(freq)

    # -------------------------
    # Download logic
    # -------------------------

    def download_partition(
        self,
        *,
        session: requests.Session,
        base_soql: str,
        window: Window,
        limit: int,
        out_file: Path,
        max_retries: int = 6,
        timeout: int = 120,
        debug: bool = False,
    ) -> int:
        """
        Download all rows for a single window to out_file (CSV), paginating with LIMIT/OFFSET.
        Returns number of rows written.
        """
        out_file.parent.mkdir(parents=True, exist_ok=True)

        predicate = self.soql_time_predicate(window.start, window.end)
        base_q = self.inject_time_filter(base_soql, predicate)

        offset = 0
        total = 0
        first_write = True

        while True:
            soql = f"{base_q} LIMIT {limit} OFFSET {offset}"

            if debug and offset == 0:
                print("SOQL (first page):", soql)

            resp = request_with_retry(
                session,
                "GET",
                self.base_url,
                params={"$query": soql},
                timeout=timeout,
                max_retries=max_retries,
                backoff_base=1.0,
                backoff_cap=60.0,
            )

            rows = resp.json()
            if not rows:
                break

            df = pd.DataFrame(rows)

            df.to_csv(
                out_file,
                mode="w" if first_write else "a",
                index=False,
                header=first_write,
            )

            n = len(df)
            total += n
            offset += n
            first_write = False

            print(
                f"    {out_file.name}: fetched {n} rows (offset={offset}, total={total})"
            )

            if n < limit:
                break

        return total


# ---------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query-file",
        default="./queries/nyc_311_noise.soql",
        help="Path to SoQL file (no LIMIT/OFFSET needed)",
    )
    ap.add_argument("--out-dir", default=OUT_DIR, help="Base output directory")
    ap.add_argument("--limit", type=int, default=1000, help="Pagination page size")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument(
        "--end-date", required=True, help="YYYY-MM-DD exclusive boundary (recommended)"
    )
    ap.add_argument(
        "--freq",
        choices=["day", "week", "month"],
        default="month",
        help="Partition frequency",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip partitions whose output file already exists",
    )
    ap.add_argument(
        "--max-retries", type=int, default=6, help="Max retries per request"
    )
    ap.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    ap.add_argument(
        "--debug", action="store_true", help="Print first-page SoQL for each partition"
    )
    args = ap.parse_args()

    load_dotenv(find_dotenv())

    dl = NYC311NoiseDownloader(
        query_file=args.query_file,
        out_dir=args.out_dir,
        freq=args.freq,
        limit=args.limit,
        resume=args.resume,
        max_retries=args.max_retries,
        timeout=args.timeout,
        debug=args.debug,
        # app_token=None -> will use NYC_APP_TOKEN env if present
    )

    dl.run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
