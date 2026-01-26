#!/usr/bin/env python3
"""
build_puma_nta_lookup.py

Build a PUMA ↔ NTA lookup table from GeoJSON polygon layers.

Usable from:
- Jupyter notebooks (import class and call .run())
- CLI (python build_puma_nta_lookup.py ...)

Method:
- Polygon overlay intersection (PUMA ∩ NTA)
- Computes intersection area shares:
    - area_share_of_puma = intersection_area / puma_area
    - area_share_of_nta  = intersection_area / nta_area
- Filters tiny overlaps via min-area-share thresholds

Outputs:
- puma_nta_lookup.(csv|parquet)

Notes:
- Uses EPSG:2263 for area (NYC local projected CRS) to make area-based shares meaningful.
- This is a geometric crosswalk. If you later obtain an official population-weighted crosswalk,
  prefer that for statistical aggregation.

Example (CLI):
  python build_puma_nta_lookup.py \
    --puma-geojson data/processed/geographies/nyc_pumas_2020.geojson \
    --nta-geojson  data/processed/geographies/nyc_ntas_2020.geojson \
    --out-dir data/processed/lookup \
    --format csv --format parquet

Example (Notebook):
  from build_puma_nta_lookup import PumaNtaLookupBuilder
  b = PumaNtaLookupBuilder(puma_geojson="...", nta_geojson="...", out_dir="data/processed/lookup")
  lookup = b.run()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import pandas as pd

DEFAULT_PUMA_GEOJSON_PATH = "./data/raw/nyc/geographies/nyc_pumas_2020.geojson"
DEFUALT_NTA_GEOJSON_PATH = "./data/raw/nyc/geographies/nyc_ntas_2020.geojson"
DEFAULT_OUTPUT_LOOKUP_FOLDER = "./data/processed/lookup"


@dataclass
class PumaNtaLookupBuilder:
    puma_geojson: Union[str, Path]
    nta_geojson: Union[str, Path]
    out_dir: Union[str, Path] = "data/processed/lookup"
    out_stem: str = "puma_nta_lookup"
    puma_id_col: Optional[str] = None
    puma_name_col: Optional[str] = None
    nta_id_col: Optional[str] = None
    nta_name_col: Optional[str] = None
    min_area_share_of_puma: float = 0.01
    min_area_share_of_nta: float = 0.01
    formats: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.puma_geojson = Path(self.puma_geojson)
        self.nta_geojson = Path(self.nta_geojson)
        self.out_dir = Path(self.out_dir)
        self.formats = self.formats or ["parquet"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        pumas = gpd.read_file(self.puma_geojson)
        ntas = gpd.read_file(self.nta_geojson)

        if len(pumas) == 0:
            raise ValueError(f"No PUMA features found in: {self.puma_geojson}")
        if len(ntas) == 0:
            raise ValueError(f"No NTA features found in: {self.nta_geojson}")

        p_id = self._pick_puma_id_col(pumas, self.puma_id_col)
        n_id = self._pick_nta_id_col(ntas, self.nta_id_col)

        # Optional names
        p_name = self.puma_name_col if self.puma_name_col in pumas.columns else None
        n_name = self._pick_nta_name_col(ntas, self.nta_name_col)

        # Keep and rename
        p_keep = [p_id, "geometry"] + ([p_name] if p_name else [])
        n_keep = [n_id, "geometry"]
        if n_name:
            n_keep.append(n_name)

        p = pumas[p_keep].rename(columns={p_id: "puma"}).copy()
        n = ntas[n_keep].rename(columns={n_id: "nta"}).copy()
        if p_name:
            p = p.rename(columns={p_name: "puma_name"})
        if n_name:
            n = n.rename(columns={n_name: "nta_name"})

        # Project for area calculations
        p_proj = p.to_crs("EPSG:2263")
        n_proj = n.to_crs("EPSG:2263")

        # Areas
        p_area = p_proj[["puma", "geometry"]].copy()
        p_area["puma_area"] = p_area.geometry.area
        p_area = p_area.drop(columns=["geometry"])

        n_area = n_proj[["nta", "geometry"]].copy()
        n_area["nta_area"] = n_area.geometry.area
        n_area = n_area.drop(columns=["geometry"])

        # Intersection overlay
        inter = gpd.overlay(p_proj, n_proj, how="intersection")
        if len(inter) == 0:
            raise RuntimeError(
                "Overlay produced zero intersections. Check geometries/CRS."
            )

        inter["intersection_area"] = inter.geometry.area
        inter = inter.drop(columns=["geometry"])

        out = inter.merge(p_area, on="puma", how="left").merge(
            n_area, on="nta", how="left"
        )
        out["area_share_of_puma"] = out["intersection_area"] / out["puma_area"]
        out["area_share_of_nta"] = out["intersection_area"] / out["nta_area"]

        # Filter small overlaps
        out = out[
            (out["area_share_of_puma"] >= float(self.min_area_share_of_puma))
            & (out["area_share_of_nta"] >= float(self.min_area_share_of_nta))
        ].copy()

        # Sort for readability
        sort_cols = ["puma", "area_share_of_puma", "nta"]
        out = out.sort_values(sort_cols, ascending=[True, False, True]).reset_index(
            drop=True
        )

        self._write(out)
        return out

    def _write(self, df: pd.DataFrame) -> None:
        out_stem = self.out_dir / self.out_stem

        if "csv" in self.formats:
            p = out_stem.with_suffix(".csv")
            df.to_csv(p, index=False)
            print(f"Wrote CSV: {p} (rows={len(df):,})")

        if "parquet" in self.formats:
            p = out_stem.with_suffix(".parquet")
            df.to_parquet(p, index=False)
            print(f"Wrote Parquet: {p} (rows={len(df):,})")

    @staticmethod
    def _pick_nta_name_col(
        gdf: gpd.GeoDataFrame, user_col: Optional[str]
    ) -> Optional[str]:
        if user_col:
            if user_col not in gdf.columns:
                raise KeyError(
                    f"--nta-name-col={user_col} not found. Columns={list(gdf.columns)}"
                )
            return user_col

        for c in ["NTAName", "NTANAME", "NAMELSAD", "nta_name"]:
            if c in gdf.columns:
                return c

        return None  # name is optional

    @staticmethod
    def _pick_puma_id_col(gdf: gpd.GeoDataFrame, user_col: Optional[str]) -> str:
        if user_col:
            if user_col not in gdf.columns:
                raise KeyError(
                    f"--puma-id-col={user_col} not found. Columns={list(gdf.columns)}"
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
            if c in gdf.columns:
                return c
        raise KeyError("Could not infer PUMA id column. Provide --puma-id-col.")

    @staticmethod
    def _pick_nta_id_col(gdf: gpd.GeoDataFrame, user_col: Optional[str]) -> str:
        if user_col:
            if user_col not in gdf.columns:
                raise KeyError(
                    f"--nta-id-col={user_col} not found. Columns={list(gdf.columns)}"
                )
            return user_col
        # Common NYC NTA fields vary by dataset; include likely candidates.
        for c in ["NTACode", "NTA2020", "nta2020", "NTA", "nta", "GEOID", "geoid"]:
            if c in gdf.columns:
                return c
        raise KeyError("Could not infer NTA id column. Provide --nta-id-col.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a PUMA↔NTA lookup table from GeoJSON polygons."
    )
    ap.add_argument(
        "--puma-geojson",
        default=DEFAULT_PUMA_GEOJSON_PATH,
        help="Path to PUMA GeoJSON.",
    )
    ap.add_argument(
        "--nta-geojson", default=DEFUALT_NTA_GEOJSON_PATH, help="Path to NTA GeoJSON."
    )
    ap.add_argument(
        "--out-dir", default=DEFAULT_OUTPUT_LOOKUP_FOLDER, help="Output directory."
    )
    ap.add_argument(
        "--out-stem", default="puma_nta_lookup", help="Output stem (no extension)."
    )
    ap.add_argument(
        "--puma-id-col", default=None, help="PUMA id column (auto-detect if omitted)."
    )
    ap.add_argument(
        "--puma-name-col", default=None, help="Optional PUMA name column to include."
    )
    ap.add_argument(
        "--nta-id-col", default=None, help="NTA id column (auto-detect if omitted)."
    )
    ap.add_argument(
        "--nta-name-col", default=None, help="Optional NTA name column to include."
    )
    ap.add_argument(
        "--min-area-share-of-puma",
        type=float,
        default=0.01,
        help="Min intersection share of PUMA.",
    )
    ap.add_argument(
        "--min-area-share-of-nta",
        type=float,
        default=0.01,
        help="Min intersection share of NTA.",
    )
    ap.add_argument(
        "--format",
        action="append",
        default="csv",
        choices=["csv", "parquet"],
        help="Output format(s). Provide one or both: --format csv --format parquet",
    )
    args = ap.parse_args()

    b = PumaNtaLookupBuilder(
        puma_geojson=args.puma_geojson,
        nta_geojson=args.nta_geojson,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
        puma_id_col=args.puma_id_col,
        puma_name_col=args.puma_name_col,
        nta_id_col=args.nta_id_col,
        nta_name_col=args.nta_name_col,
        min_area_share_of_puma=args.min_area_share_of_puma,
        min_area_share_of_nta=args.min_area_share_of_nta,
        formats=args.format,
    )

    _ = b.run()


if __name__ == "__main__":
    main()
