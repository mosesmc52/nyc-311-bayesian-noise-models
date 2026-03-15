#!/usr/bin/env python3
"""
build_noise_counts_with_lookup.py

Build noise complaint counts by PUMA (via point-in-polygon), then optionally
map/aggregate to NTA using a pre-built PUMA↔NTA lookup table.

Key changes vs prior version:
- Removes neighborhood polygon overlay logic
- Adds lookup-table join (PUMA→NTA) with optional weighting
- Supports reading a folder of CSVs and summing across all files
- Keeps both notebook-friendly class usage and CLI

Typical flow:
1) complaints CSV(s) -> spatial join -> PUMA
2) aggregate counts by PUMA (and other dimensions)
3) if lookup provided:
      - join PUMA→NTA (1-to-many allowed)
      - compute weighted_count = complaint_count * weight
      - aggregate to NTA (or keep expanded rows)

Outputs:
- always writes the base PUMA-level counts table
- optionally writes an NTA-level counts table (if --lookup-csv and --geo-level nta)

Examples:
  # PUMA only
  python scripts/features/build_noise_counts_with_lookup.py \
    --noise-csv data/raw/nyc/311_noise/by_month \
    --puma-geojson data/processed/geographies/nyc_pumas_2020.geojson \
    --format parquet \
    --geo-level puma

  # Aggregate to NTA using lookup
  python scripts/features/build_noise_counts_with_lookup.py \
    --noise-csv data/raw/nyc/311_noise/by_month \
    --puma-geojson data/processed/geographies/nyc_pumas_2020.geojson \
    --lookup-csv data/processed/lookup/puma_nta_lookup.parquet \
    --lookup-weight-col area_share_of_puma \
    --format parquet \
    --geo-level nta

Notes on weights:
- If your lookup was built from polygon intersections, a reasonable default weight is
  area_share_of_puma (intersection_area / puma_area).
- For “public” reporting and modeling, population-weighted crosswalks are preferable
  if you have them. This script supports any weight column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd

DEFAULT_NOISE_CSV_PATH = "./data/raw/nyc/311_noise/by_month"
DEFAULT_PUMA_GEOJSON_PATH = "./data/raw/nyc/geographies/nyc_pumas_2020.geojson"
DEFUALT_NTA_GEOJSON_PATH = "./data/raw/nyc/geographies/nyc_ntas_2020.geojson"
DEFAULT_LOOKUP_CSV = "./data/processed/lookup/puma_nta_lookup.parquet"
OUTPUT_DIR = "data/processed/features"


class PumaNoiseCountsWithLookup:
    def __init__(
        self,
        *,
        puma_geojson: Union[str, Path],
        out_dir: Union[str, Path] = "data/processed/features",
        out_stem_puma: str = "puma_noise_counts",
        out_stem_geo: str = "geo_noise_counts",
        puma_id_col: Optional[str] = None,
        puma_name_col: Optional[str] = None,
        predicate: str = "within",
        time_bucket: str = "day",
        tod_scheme: str = "none",
        tod_cuts: Optional[str] = None,
        tod_labels: Optional[str] = None,
        drop_unmatched: bool = False,
        formats: Optional[List[str]] = None,
        # Lookup mapping (optional)
        lookup_csv: Optional[Union[str, Path]] = None,
        lookup_puma_col: str = "puma",
        lookup_geo_col: str = "nta",
        lookup_geo_name_col: Optional[str] = "nta_name",
        lookup_weight_col: Optional[str] = None,  # e.g., area_share_of_puma
        geo_level: str = "puma",  # "puma" or "nta" (or any lookup_geo_col semantics)
    ) -> None:
        self.puma_geojson = Path(puma_geojson)
        self.out_dir = Path(out_dir)
        self.out_stem_puma = out_stem_puma
        self.out_stem_geo = out_stem_geo
        self.puma_id_col = puma_id_col
        self.puma_name_col = puma_name_col
        self.predicate = predicate
        self.time_bucket = time_bucket
        self.tod_scheme = tod_scheme
        self.tod_cuts = tod_cuts
        self.tod_labels = tod_labels
        self.drop_unmatched = drop_unmatched
        self.formats = formats or ["parquet"]

        self.lookup_csv = Path(lookup_csv) if lookup_csv else None
        self.lookup_puma_col = lookup_puma_col
        self.lookup_geo_col = lookup_geo_col
        self.lookup_geo_name_col = lookup_geo_name_col
        self.lookup_weight_col = lookup_weight_col
        self.geo_level = geo_level.lower()

        self._validate()

    # ----------------------------
    # Public API
    # ----------------------------

    def _build_puma_primary_nta(self, lookup: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce a PUMA↔NTA lookup to one row per PUMA by selecting
        the NTA with the largest weight (area_share_of_puma or similar).
        """
        if self.lookup_weight_col is None:
            raise ValueError(
                "Including NTA in puma_noise_counts requires lookup_weight_col "
                "(e.g. area_share_of_puma) to choose a primary NTA."
            )

        required = {"puma", "nta", self.lookup_weight_col}
        missing = required - set(lookup.columns)
        if missing:
            raise KeyError(f"Lookup missing required columns: {missing}")

        cols = ["puma", "nta", self.lookup_weight_col]
        if self.lookup_geo_name_col and self.lookup_geo_name_col in lookup.columns:
            cols.append(self.lookup_geo_name_col)

        lk = lookup[cols].copy()
        lk["puma"] = lk["puma"].astype("string").str.strip()
        lk["nta"] = lk["nta"].astype("string").str.strip()
        if self.lookup_geo_name_col and self.lookup_geo_name_col in lk.columns:
            lk[self.lookup_geo_name_col] = (
                lk[self.lookup_geo_name_col].astype("string").str.strip()
            )

        # Pick the dominant NTA per PUMA
        idx = lk.groupby("puma")[self.lookup_weight_col].idxmax()
        primary = lk.loc[idx].reset_index(drop=True)

        primary = primary.rename(
            columns={
                "nta": "nta",
                self.lookup_geo_name_col: (
                    "nta_name" if self.lookup_geo_name_col else None
                ),
            }
        )

        return primary

    def run(self, noise_csv: Union[str, Path]) -> pd.DataFrame:
        """
        Always writes/returns PUMA-level counts.
        If geo_level != "puma" and lookup_csv provided, also writes geo-level counts.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        df = self._read_noise_inputs(Path(noise_csv))
        df = self._normalize_columns(df)

        pumas = self._read_pumas(self.puma_geojson)
        pumas = self._prepare_pumas(pumas)

        # complaints points
        complaints = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )

        # point -> puma
        joined = gpd.sjoin(complaints, pumas, how="left", predicate=self.predicate)
        if self.drop_unmatched:
            joined = joined.dropna(subset=["puma"])

        # time bucketing
        joined["created_bucket"] = self._time_bucket(
            joined["created_date"], self.time_bucket
        )

        # TOD labeling (optional)
        if self.tod_scheme != "none":
            joined["time_of_day"] = self._add_time_of_day_period(
                joined,
                scheme=self.tod_scheme,
                cuts_str=self.tod_cuts,
                labels_str=self.tod_labels,
            )

        # base aggregation at PUMA
        puma_group_cols: List[str] = [
            "puma",
            "complaint_type",
            "descriptor",
            "location_type",
            "created_bucket",
        ]
        if "puma_name" in joined.columns:
            puma_group_cols.insert(1, "puma_name")
        if self.tod_scheme != "none":
            puma_group_cols.append("time_of_day")

        puma_counts = (
            joined.groupby(puma_group_cols, dropna=False)
            .size()
            .reset_index(name="complaint_count")
            .sort_values(puma_group_cols)
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Attach NTA labels to PUMA counts (descriptive only)
        # -------------------------------------------------
        if self.lookup_csv is not None:
            lookup = self._read_lookup(self.lookup_csv)
            primary_nta = self._build_puma_primary_nta(lookup)
            puma_counts["puma"] = puma_counts["puma"].astype("string").str.strip()

            puma_counts = puma_counts.merge(
                primary_nta,
                on="puma",
                how="left",
            )

        # write puma counts
        self._write_outputs(puma_counts, self.out_dir / self.out_stem_puma)

        # diagnostics
        unmatched = int(joined["puma"].isna().sum())
        total = int(len(joined))
        print(
            f"Diagnostics: matched={(total - unmatched):,} unmatched={unmatched:,} total={total:,}"
        )

        # optionally map/aggregate to geo level via lookup
        if self.geo_level != "puma":
            if self.lookup_csv is None:
                raise ValueError("--geo-level != puma requires --lookup-csv")

            geo_counts = self._aggregate_via_lookup(puma_counts)
            self._write_outputs(geo_counts, self.out_dir / self.out_stem_geo)
            return geo_counts

        return puma_counts

    # ----------------------------
    # Lookup aggregation
    # ----------------------------

    def _aggregate_via_lookup(self, puma_counts: pd.DataFrame) -> pd.DataFrame:
        lookup = self._read_lookup(self.lookup_csv)

        # join on puma -> geo
        if self.lookup_puma_col not in lookup.columns:
            raise KeyError(
                f"lookup missing puma column '{self.lookup_puma_col}'. Columns={list(lookup.columns)}"
            )
        if self.lookup_geo_col not in lookup.columns:
            raise KeyError(
                f"lookup missing geo column '{self.lookup_geo_col}'. Columns={list(lookup.columns)}"
            )

        # Keep only needed lookup cols
        keep = [self.lookup_puma_col, self.lookup_geo_col]
        if self.lookup_geo_name_col and self.lookup_geo_name_col in lookup.columns:
            keep.append(self.lookup_geo_name_col)
        if self.lookup_weight_col:
            if self.lookup_weight_col not in lookup.columns:
                raise KeyError(
                    f"lookup missing weight column '{self.lookup_weight_col}'. Columns={list(lookup.columns)}"
                )
            keep.append(self.lookup_weight_col)

        lk = lookup[keep].copy()

        # Standardize columns for merge
        lk = lk.rename(
            columns={
                self.lookup_puma_col: "puma",
                self.lookup_geo_col: "geo_id",
                **(
                    {self.lookup_geo_name_col: "geo_name"}
                    if self.lookup_geo_name_col
                    and self.lookup_geo_name_col in lk.columns
                    else {}
                ),
                **(
                    {self.lookup_weight_col: "weight"} if self.lookup_weight_col else {}
                ),
            }
        )

        expanded = puma_counts.merge(lk, on="puma", how="left")

        # If no weight provided, treat as 1.0 (note: if lookup is 1-to-many, you probably DO want weights)
        if "weight" not in expanded.columns:
            expanded["weight"] = 1.0

        expanded["weighted_count"] = expanded["complaint_count"] * expanded["weight"]

        # Geo grouping: replace puma with geo_id (+ optional geo_name)
        geo_group_cols = [
            "geo_id",
            "complaint_type",
            "descriptor",
            "location_type",
            "created_bucket",
        ]
        if "geo_name" in expanded.columns:
            geo_group_cols.insert(1, "geo_name")
        if self.tod_scheme != "none" and "time_of_day" in expanded.columns:
            geo_group_cols.append("time_of_day")

        geo_counts = (
            expanded.groupby(geo_group_cols, dropna=False)["weighted_count"]
            .sum()
            .reset_index()
            .rename(columns={"weighted_count": "complaint_count"})
            .sort_values(geo_group_cols)
            .reset_index(drop=True)
        )

        return geo_counts

    @staticmethod
    def _read_lookup(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"lookup file not found: {path}")

        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)

        raise ValueError(f"lookup must be .csv or .parquet, got: {path.suffix}")

    # ----------------------------
    # IO helpers
    # ----------------------------

    @staticmethod
    def _read_noise_inputs(path: Path) -> pd.DataFrame:
        if path.is_file():
            return pd.read_csv(path)

        if path.is_dir():
            files = sorted(path.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No .csv files found in directory: {path}")
            dfs = []
            for f in files:
                print(f"Reading: {f}")
                usecols = [
                    "latitude",
                    "longitude",
                    "created_date",
                    "complaint_type",
                    "descriptor",
                    "location_type",
                ]
                dfs.append(pd.read_csv(f, usecols=usecols, low_memory=False))
            return pd.concat(dfs, ignore_index=True)

        raise FileNotFoundError(f"noise_csv path not found: {path}")

    def _write_outputs(self, df: pd.DataFrame, out_stem: Path) -> None:
        if "csv" in self.formats:
            p = out_stem.with_suffix(".csv")
            df.to_csv(p, index=False)
            print(f"Wrote CSV: {p} (rows={len(df):,})")
        if "parquet" in self.formats:
            p = out_stem.with_suffix(".parquet")
            df.to_parquet(p, index=False)
            print(f"Wrote Parquet: {p} (rows={len(df):,})")

    # ----------------------------
    # PUMA helpers
    # ----------------------------

    @staticmethod
    def _read_pumas(path: Path) -> gpd.GeoDataFrame:
        if path.is_dir():
            files = sorted(list(path.glob("*.geojson"))) + sorted(
                list(path.glob("*.json"))
            )
            if not files:
                raise FileNotFoundError(
                    f"No .geojson/.json files found in directory: {path}"
                )
            gdfs = []
            for f in files:
                g = gpd.read_file(f)
                if len(g) == 0:
                    continue
                gdfs.append(g)
            if not gdfs:
                raise RuntimeError(f"All PUMA files were empty in: {path}")
            pumas = pd.concat(gdfs, ignore_index=True)
            return gpd.GeoDataFrame(pumas, geometry="geometry")
        if path.is_file():
            return gpd.read_file(path)
        raise FileNotFoundError(f"PUMA path not found: {path}")

    @staticmethod
    def _pick_puma_id_column(pumas: gpd.GeoDataFrame, user_col: Optional[str]) -> str:
        if user_col:
            if user_col not in pumas.columns:
                raise KeyError(
                    f"--puma-id-col={user_col} not found. Columns={list(pumas.columns)}"
                )
            return user_col
        for c in [
            "PUMACE20",
            "PUMACE10",
            "pumace20",
            "pumace10",
            "PUMA",
            "puma",
            "GEOID20",
            "GEOID10",
            "GEOID",
            "geoid",
        ]:
            if c in pumas.columns:
                return c
        raise KeyError("Could not infer PUMA id column. Provide --puma-id-col.")

    def _prepare_pumas(self, pumas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if pumas.crs is None:
            pumas = pumas.set_crs("EPSG:4326", allow_override=True)
        else:
            pumas = pumas.to_crs("EPSG:4326")

        pid = self._pick_puma_id_column(pumas, self.puma_id_col)

        keep = [pid, "geometry"]
        if self.puma_name_col:
            if self.puma_name_col not in pumas.columns:
                raise KeyError(
                    f"--puma-name-col={self.puma_name_col} not found. Columns={list(pumas.columns)}"
                )
            keep.insert(1, self.puma_name_col)

        out = pumas[keep].copy().rename(columns={pid: "puma"})
        out["puma"] = out["puma"].astype("string").str.strip()

        if self.puma_name_col:
            out = out.rename(columns={self.puma_name_col: "puma_name"})
        return out

    # ----------------------------
    # Noise schema + time utilities
    # ----------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "latitude",
            "longitude",
            "created_date",
            "complaint_type",
            "descriptor",
            "location_type",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Noise CSV missing required columns: {missing}. Found: {list(df.columns)}"
            )

        df = df.copy()
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")

        df = df.dropna(subset=["created_date", "latitude", "longitude"])
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        return df

    @staticmethod
    def _time_bucket(series: pd.Series, bucket: str) -> pd.Series:
        if bucket == "day":
            return series.dt.floor("D")
        if bucket == "hour":
            return series.dt.floor("H")
        if bucket == "timestamp":
            return series
        raise ValueError("--time-bucket must be one of: day, hour, timestamp")

    @staticmethod
    def _parse_cuts(s: str) -> List[int]:
        cuts = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        cuts = sorted(cuts)
        if not cuts or cuts[0] != 0 or cuts[-1] != 24:
            raise ValueError(
                "--tod-cuts must start at 0 and end at 24 (e.g., 0,6,12,18,24)"
            )
        if len(set(cuts)) != len(cuts):
            raise ValueError("--tod-cuts must not contain duplicates")
        if any(c < 0 or c > 24 for c in cuts):
            raise ValueError("--tod-cuts hours must be in [0, 24]")
        return cuts

    @staticmethod
    def _parse_labels(s: str) -> List[str]:
        labels = [x.strip() for x in s.split(",") if x.strip() != ""]
        if not labels:
            raise ValueError("--tod-labels provided but empty")
        return labels

    @staticmethod
    def _default_range_labels(cuts: List[int]) -> List[str]:
        return [f"{cuts[i]:02d}-{cuts[i+1]:02d}" for i in range(len(cuts) - 1)]

    def _tod_bins_from_scheme(
        self,
        scheme: str,
        cuts_str: Optional[str],
        labels_str: Optional[str],
    ) -> tuple[list[int], list[str]]:
        scheme = scheme.lower()

        if scheme == "none":
            return [], []
        if scheme == "two":
            # Night: 8pm–6am, Day: 6am–8pm
            cuts = [0, 6, 20, 24]
            labels = ["night", "day", "night"]
            return cuts, labels
        elif scheme == "four":
            cuts = [0, 6, 12, 18, 24]
        elif scheme == "six":
            cuts = [0, 4, 8, 12, 16, 20, 24]
        elif scheme == "nightlife":
            cuts = [0, 3, 6, 12, 18, 22, 24]
        elif scheme == "custom":
            if not cuts_str:
                raise ValueError(
                    "--tod-scheme custom requires --tod-cuts, e.g. 0,6,12,18,24"
                )
            cuts = self._parse_cuts(cuts_str)
        else:
            raise ValueError(
                "--tod-scheme must be one of: none, four, six, nightlife, custom"
            )

        if labels_str:
            labels = self._parse_labels(labels_str)
            expected = len(cuts) - 1
            if len(labels) != expected:
                raise ValueError(
                    f"--tod-labels count ({len(labels)}) must equal intervals ({expected})"
                )
        else:
            labels = self._default_range_labels(cuts)

        return cuts, labels

    def _add_time_of_day_period(
        self,
        df: pd.DataFrame,
        *,
        scheme: str,
        cuts_str: Optional[str],
        labels_str: Optional[str],
    ) -> pd.Series:
        cuts, labels = self._tod_bins_from_scheme(scheme, cuts_str, labels_str)
        if not cuts:
            return pd.Series([None] * len(df), index=df.index, dtype="object")
        hours = pd.to_datetime(df["created_date"], errors="coerce").dt.hour
        ordered = True
        if len(labels) != len(set(labels)):
            ordered = False  # allow duplicate labels like ["night","day","night"]

        cat = pd.cut(
            hours,
            bins=cuts,
            right=False,
            labels=labels,
            include_lowest=True,
            ordered=ordered,
        )
        return cat.astype("string")

    # ----------------------------
    # Validation
    # ----------------------------

    def _validate(self) -> None:
        if self.predicate not in ("within", "intersects"):
            raise ValueError("--predicate must be one of: within, intersects")
        if self.time_bucket not in ("day", "hour", "timestamp"):
            raise ValueError("--time-bucket must be one of: day, hour, timestamp")
        if self.tod_scheme not in ("none", "two", "four", "six", "nightlife", "custom"):
            raise ValueError(
                "--tod-scheme must be one of: none, two, four, six, nightlife, custom"
            )

        bad = [f for f in self.formats if f not in ("csv", "parquet")]
        if bad:
            raise ValueError(f"Invalid format(s): {bad}. Use csv and/or parquet.")

        if self.geo_level not in ("puma", "nta"):
            raise ValueError("--geo-level must be one of: puma, nta")

        if self.geo_level != "puma" and self.lookup_csv is None:
            raise ValueError("--geo-level nta requires --lookup-csv")


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--noise-csv",
        default=DEFAULT_NOISE_CSV_PATH,
        help="CSV file OR directory of CSVs.",
    )
    ap.add_argument(
        "--puma-geojson",
        default=DEFAULT_PUMA_GEOJSON_PATH,
        help="PUMA GeoJSON file OR directory of per-PUMA files.",
    )
    ap.add_argument("--out-dir", default=OUTPUT_DIR, help="Output directory.")
    ap.add_argument(
        "--out-stem-puma",
        default="puma_noise_counts",
        help="Output stem for PUMA counts.",
    )
    ap.add_argument(
        "--out-stem-geo",
        default="nta_noise_counts",
        help="Output stem for geo-level counts (e.g., NTA).",
    )
    ap.add_argument(
        "--format",
        default=["csv", "parquet"],
        action="append",
        choices=["csv", "parquet"],
        help="Output format(s). Provide one or both: --format csv --format parquet",
    )
    ap.add_argument(
        "--puma-id-col", default=None, help="PUMA id column (auto-detect if omitted)."
    )
    ap.add_argument("--puma-name-col", default=None, help="Optional PUMA name column.")
    ap.add_argument(
        "--predicate",
        default="within",
        choices=["within", "intersects"],
        help="Spatial join predicate.",
    )
    ap.add_argument(
        "--time-bucket",
        default="day",
        choices=["day", "hour", "timestamp"],
        help="created_date bucket.",
    )
    ap.add_argument(
        "--tod-scheme",
        default="two",
        choices=["none", "two", "four", "six", "nightlife", "custom"],
    )
    ap.add_argument(
        "--tod-cuts",
        default=None,
        help="Custom cuts, e.g. 0,6,12,18,24 (required if tod-scheme=custom).",
    )
    ap.add_argument(
        "--tod-labels",
        default=None,
        help="Custom labels, e.g. '00-06,06-12,12-18,18-24'.",
    )
    ap.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="Drop complaints not matched to a PUMA.",
    )

    # Lookup mapping
    ap.add_argument(
        "--geo-level",
        default="puma",
        choices=["puma", "nta"],
        help="Output aggregation geography.",
    )
    ap.add_argument(
        "--lookup-csv",
        default=DEFAULT_LOOKUP_CSV,
        help="PUMA→NTA lookup table (.csv or .parquet). Required for nta.",
    )
    ap.add_argument(
        "--lookup-puma-col", default="puma", help="Lookup column name for PUMA id."
    )
    ap.add_argument(
        "--lookup-geo-col",
        default="nta",
        help="Lookup column name for geo id (e.g., NTA code).",
    )
    ap.add_argument(
        "--lookup-geo-name-col",
        default="nta_name",
        help="Optional lookup column for geo name.",
    )
    ap.add_argument(
        "--lookup-weight-col",
        default="area_share_of_puma",
        help="Optional lookup weight column (e.g., area_share_of_puma). If omitted, weight=1.0.",
    )

    args = ap.parse_args()

    runner = PumaNoiseCountsWithLookup(
        puma_geojson=args.puma_geojson,
        out_dir=args.out_dir,
        out_stem_puma=args.out_stem_puma,
        out_stem_geo=args.out_stem_geo,
        puma_id_col=args.puma_id_col,
        puma_name_col=args.puma_name_col,
        predicate=args.predicate,
        time_bucket=args.time_bucket,
        tod_scheme=args.tod_scheme,
        tod_cuts=args.tod_cuts,
        tod_labels=args.tod_labels,
        drop_unmatched=args.drop_unmatched,
        formats=args.format,
        lookup_csv=args.lookup_csv,
        lookup_puma_col=args.lookup_puma_col,
        lookup_geo_col=args.lookup_geo_col,
        lookup_geo_name_col=args.lookup_geo_name_col,
        lookup_weight_col=args.lookup_weight_col,
        geo_level=args.geo_level,
    )

    _ = runner.run(args.noise_csv)


if __name__ == "__main__":
    main()
