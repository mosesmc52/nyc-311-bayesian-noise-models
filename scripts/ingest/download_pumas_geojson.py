#!/usr/bin/env python3
"""
Download NYC PUMA GeoJSON (ArcGIS REST) into a user-specified folder.

Default behavior:
- Downloads the full FeatureCollection and writes one file:
  <out_dir>/nyc_pumas_2020.geojson

Optional:
- --split writes one file per feature (per PUMA) into:
  <out_dir>/pumas/<PUMA_ID>.geojson
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

DEFAULT_URL = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/"
    "NYC_Public_Use_Microdata_Areas_PUMAs_2020/FeatureServer/0/query"
    "?where=1=1&outFields=*&outSR=4326&f=pgeojson"
)

OUT_DIR = "./data/raw/nyc/geographies"


@dataclass
class PumaGeoJSONDownloader:
    """
    Utility class that downloads a GeoJSON FeatureCollection and saves it
    either as a single file or split into one file per feature.
    """

    out_dir: Path = OUT_DIR
    url: str = DEFAULT_URL
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
        filename: str = "nyc_pumas_2020.geojson",
        id_field: Optional[str] = None,
        name_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch GeoJSON from self.url and write it to disk.

        Returns the parsed GeoJSON dict.

        Args:
            split: If True, writes one GeoJSON per feature under <out_dir>/pumas/.
            filename: Combined output filename (ignored if split=True).
            id_field: Property field used for per-feature filenames (auto-detect if None).
            name_field: Optional property field appended to per-feature filenames.
        """
        fc = self._fetch_json()

        if split:
            n = self._split_features(fc, id_field=id_field, name_field=name_field)
            print(f"Wrote {n} per-feature GeoJSON files to: {self.out_dir / 'pumas'}")
        else:
            out_path = self.out_dir / filename
            self._write_geojson(out_path, fc)
            feats = fc.get("features")
            n = len(feats) if isinstance(feats, list) else None
            msg = f"Wrote combined GeoJSON to: {out_path}"
            if n is not None:
                msg += f"  (features={n})"
            print(msg)

        return fc

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

                # ArcGIS errors can come back as JSON with an "error" key
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"ArcGIS error payload: {data.get('error')}")

                return data
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.backoff**attempt)
                else:
                    break

        raise RuntimeError(
            f"Failed to fetch GeoJSON after {self.retries} attempts: {last_err}"
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
        id_field: Optional[str] = None,
        name_field: Optional[str] = None,
    ) -> int:
        if fc.get("type") != "FeatureCollection":
            raise ValueError(f"Expected FeatureCollection, got type={fc.get('type')}")

        features = fc.get("features") or []
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("No features found in GeoJSON (features list is empty).")

        # Auto-detect id_field if not provided
        if not id_field:
            candidates = [
                "PUMA",
                "puma",
                "PUMACE20",
                "pumace20",
                "PUMA20",
                "puma20",
                "GEOID",
                "geoid",
            ]
            props0 = (
                (features[0].get("properties") or {})
                if isinstance(features[0], dict)
                else {}
            )
            for c in candidates:
                if c in props0:
                    id_field = c
                    break

        split_dir = self.out_dir / "pumas"
        split_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for feat in features:
            if not isinstance(feat, dict):
                continue
            props = feat.get("properties") or {}

            fid = None
            if id_field and id_field in props:
                fid = props.get(id_field)

            if fid is None:
                fid = f"feature_{n:04d}"

            base = self._safe_slug(str(fid))

            if name_field and name_field in props and props.get(name_field):
                base = f"{base}__{self._safe_slug(str(props.get(name_field)))}"

            out_path = split_dir / f"{base}.geojson"

            single_fc = {
                "type": "FeatureCollection",
                "features": [feat],
                # carry through other top-level keys (if any) except features/type
                **{k: v for k, v in fc.items() if k not in ("features", "type")},
            }
            self._write_geojson(out_path, single_fc)
            n += 1

        return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Download NYC PUMA GeoJSON (ArcGIS REST).")
    ap.add_argument(
        "--out-dir",
        default=OUT_DIR,
        required=False,
        help="Output folder (created if missing).",
    )
    ap.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="ArcGIS REST endpoint returning GeoJSON.",
    )
    ap.add_argument(
        "--filename",
        default="nyc_pumas_2020.geojson",
        help="Output filename for combined GeoJSON (ignored if --split).",
    )
    ap.add_argument(
        "--split",
        action="store_true",
        help="Write one GeoJSON per feature under <out-dir>/pumas/.",
    )
    ap.add_argument(
        "--id-field",
        default=None,
        help="Property field used for per-feature filenames (auto-detect if omitted).",
    )
    ap.add_argument(
        "--name-field",
        default=None,
        help="Optional property field appended to per-feature filenames.",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    ap.add_argument("--retries", type=int, default=3, help="Number of retries.")
    ap.add_argument(
        "--backoff", type=float, default=1.5, help="Backoff base for retries."
    )
    args = ap.parse_args()

    dl = PumaGeoJSONDownloader(
        out_dir=Path(args.out_dir),
        url=args.url,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )

    dl.download(
        split=args.split,
        filename=args.filename,
        id_field=args.id_field,
        name_field=args.name_field,
    )


if __name__ == "__main__":
    main()
