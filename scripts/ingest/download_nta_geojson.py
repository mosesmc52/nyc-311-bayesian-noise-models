#!/usr/bin/env python3
"""
download_nta_geojson.py

Download NYC Neighborhood Tabulation Areas (NTA) 2020 GeoJSON
from NYC Planning / ArcGIS.

Usable from:
- Jupyter notebooks (import the class and call .download())
- Command line (python download_nta_geojson.py ...)

Default behavior:
- Writes a single combined GeoJSON:
    <out_dir>/nyc_ntas_2020.geojson

Optional:
- --split writes one GeoJSON per NTA under:
    <out_dir>/ntas/<NTA_ID>.geojson
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

NTA_GEOJSON_URL = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/"
    "NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0/query"
    "?where=1=1&outFields=*&outSR=4326&f=pgeojson"
)
DEFAULT_OUT_DIR = "./data/raw/nyc/geographies"
DEFAULT_FILENAME = "nyc_ntas_2020.geojson"


@dataclass
class NtaGeoJSONDownloader:
    """
    Downloader for NYC Neighborhood Tabulation Areas (NTA) GeoJSON.
    """

    out_dir: Path
    url: str = NTA_GEOJSON_URL
    timeout: int = 60
    retries: int = 3
    backoff: float = 1.5

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Public API
    # -------------------------

    def download(
        self,
        *,
        split: bool = False,
        filename: str = DEFAULT_FILENAME,
        id_field: Optional[str] = None,
        name_field: Optional[str] = None,
        return_data: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Download NTA GeoJSON and write to disk.

        Args:
            split: If True, write one GeoJSON per NTA under <out_dir>/ntas/.
            filename: Output filename for combined mode.
            id_field: Property field used for per-feature filenames (auto-detect if None).
            name_field: Optional property appended to per-feature filenames.
            return_data: If True, return parsed GeoJSON dict.

        Returns:
            Parsed GeoJSON dict if return_data=True, else None.
        """
        fc = self._fetch_json()

        if split:
            split_dir = self.out_dir / "ntas"
            n = self._split_features(
                fc,
                split_dir=split_dir,
                id_field=id_field,
                name_field=name_field,
            )
            print(f"Wrote {n} per-NTA GeoJSON files to: {split_dir}")
        else:
            out_path = self.out_dir / filename
            self._write_geojson(out_path, fc)
            n = len(fc.get("features", []))
            print(f"Wrote combined GeoJSON to: {out_path}  (features={n})")

        return fc if return_data else None

    # -------------------------
    # Internals
    # -------------------------

    def _fetch_json(self) -> Dict[str, Any]:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(self.url, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()

                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"ArcGIS error payload: {data.get('error')}")

                return data

            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff**attempt)

        raise RuntimeError(
            f"Failed to fetch NTA GeoJSON after {self.retries} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _write_geojson(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _safe_slug(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "unknown"

    def _split_features(
        self,
        fc: Dict[str, Any],
        *,
        split_dir: Path,
        id_field: Optional[str],
        name_field: Optional[str],
    ) -> int:
        if fc.get("type") != "FeatureCollection":
            raise ValueError(f"Expected FeatureCollection, got {fc.get('type')}")

        features = fc.get("features", [])
        if not features:
            raise ValueError("GeoJSON contains no features.")

        # Auto-detect NTA id field if not provided
        if not id_field:
            props0 = features[0].get("properties", {})
            for c in ["NTACode", "NTA2020", "nta2020", "NTA", "nta", "GEOID"]:
                if c in props0:
                    id_field = c
                    break

        if not id_field:
            raise ValueError("Could not infer NTA id field. Provide --id-field.")

        split_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for feat in features:
            props = feat.get("properties", {})
            fid = props.get(id_field, f"feature_{n:04d}")

            base = self._safe_slug(str(fid))
            if name_field and name_field in props:
                base = f"{base}__{self._safe_slug(str(props[name_field]))}"

            out_path = split_dir / f"{base}.geojson"

            single_fc = {
                "type": "FeatureCollection",
                "features": [feat],
                **{k: v for k, v in fc.items() if k not in ("type", "features")},
            }

            self._write_geojson(out_path, single_fc)
            n += 1

        return n


# -------------------------
# CLI
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download NYC Neighborhood Tabulation Areas (NTA) GeoJSON."
    )
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    ap.add_argument(
        "--filename", default=DEFAULT_FILENAME, help="Output filename (combined mode)."
    )
    ap.add_argument("--split", action="store_true", help="Write one GeoJSON per NTA.")
    ap.add_argument(
        "--id-field", default=None, help="NTA ID property (auto-detect if omitted)."
    )
    ap.add_argument("--name-field", default=None, help="Optional NTA name property.")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    ap.add_argument("--retries", type=int, default=3, help="Retry attempts.")
    ap.add_argument("--backoff", type=float, default=1.5, help="Retry backoff base.")
    args = ap.parse_args()

    dl = NtaGeoJSONDownloader(
        out_dir=Path(args.out_dir),
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )

    dl.download(
        split=args.split,
        filename=args.filename,
        id_field=args.id_field,
        name_field=args.name_field,
        return_data=False,
    )


if __name__ == "__main__":
    main()
