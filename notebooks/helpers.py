import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import arviz as az
import matplotlib.pyplot as plt

# =========================================================
# NEW: Aggregate descriptors into higher-level categories
# =========================================================
def descriptor_group(desc: str) -> str:
    d = str(desc).strip().lower()

    # Social / Party noise
    if d in {
        "loud music/party",
        "loud talking",
        "loud television",
        "car/truck music",
    }:
        return "Social / Party"

    # Animal noise
    if d in {
        "noise, barking dog",
        "noise, other animals",
    }:
        return "Animal"

    # Construction / Industrial noise
    if d in {
        "noise: construction equipment",
        "noise: construction before/after hours",
        "noise: jack hammering",
        "noise: manufacturing noise",
    }:
        return "Construction / Industrial"

    # Mechanical / Infrastructure noise
    if d in {
        "noise: air condition/ventilation equipment",
        "noise: alarms",
        "engine idling",
        "noise: private carting noise",
        "car/truck horn",
        "banging/pounding",
    }:
        return "Mechanical Equipment"

    # Optional: Transportation / Street (if you want a separate bucket later)
    # if d in {...}:
    #     return "Transportation / Street"

    return "Other"

def prep_the_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ensure datetime ---
    df["created_bucket"] = pd.to_datetime(df["created_bucket"], errors="coerce")

    # --- clean / normalize keys ---
    df["puma"] = df["puma"].astype("string").str.strip()
    df["nta_name"] = df["nta_name"].astype("string").str.strip()

    # --- combine geography label (NTA + PUMA) ---
    df["nta_puma"] = (
        df["nta_name"].fillna("Unknown")
        + " — "
        + df["puma"].fillna("Unknown")
    )

    # --- derive calendar fields ---
    df["dow"] = df["created_bucket"].dt.day_name()
    df["month"] = df["created_bucket"].dt.month_name()

    # --- clean raw descriptor text (remove parentheses + normalize whitespace) ---
    df["descriptor"] = (
        df["descriptor"]
        .astype("string")
        .fillna("")
        .str.replace(r"\([^)]*\)", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --- weekend flag (Saturday/Sunday) ---
    df["is_weekend"] = df["dow"].isin(["Saturday", "Sunday"]).astype("int8")

    # --- month_year label ---
    df["month_year"] = (
        df["month"].astype("string")
        + "__"
        + df["created_bucket"].dt.year.astype("Int64").astype("string")
    )

    df["descriptor_group"] = df["descriptor"].map(descriptor_group).astype("string")

    # --- build dow_complaint from aggregated descriptor_group ---
    df["dow_complaint"] = (
        df["descriptor_group"]
        .astype("string")
        .str.upper()
        .str.replace(r"[,_/]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace(" ", "_")
        + "__"
        + df["dow"].astype("string")
    )

    return df





def export_puma_kepler(
    df: pd.DataFrame,
    *,
    puma_geojson_path: str,
    puma_key: str = "puma",
    geo_puma_col: str = "PUMA",
    value_cols: list[str] | None = None,
    fill_value: float | int | None = 0,
    out_path: str | Path,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Merge an aggregated dataframe with PUMA polygons and export a Kepler-ready GeoJSON.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated data with a PUMA identifier column.
    puma_geojson_path : str
        Path to PUMA GeoJSON file.
    puma_key : str, default="puma"
        Column name in df containing PUMA ids.
    geo_puma_col : str, default="PUMA"
        Column name in the GeoJSON containing PUMA ids.
    value_cols : list[str], optional
        Columns to fill missing values for (e.g. ["typical_daily_count"]).
        If None, no filling is applied.
    fill_value : float or int or None, default=0
        Value to use for filling missing entries in value_cols.
        Set to None to skip filling.
    out_path : str or Path
        Output GeoJSON path.
    crs : str, default="EPSG:4326"
        Target CRS for Kepler.gl.

    Returns
    -------
    gpd.GeoDataFrame
        Kepler-ready GeoDataFrame.
    """

    # Load geometry
    gdf_puma = gpd.read_file(puma_geojson_path)
    gdf_puma[puma_key] = gdf_puma[geo_puma_col].astype(str)

    # Align key type in data
    df = df.copy()
    df[puma_key] = df[puma_key].astype(str)

    # Merge
    gdf = gdf_puma.merge(df, on=puma_key, how="left")

    # CRS for Kepler
    if gdf.crs is None or gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)

    # Fill missing values if requested
    if value_cols and fill_value is not None:
        for col in value_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna(fill_value)

    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    gdf.to_file(out_path, driver="GeoJSON")

    return gdf


# Combined label for filtering (complaint__dow)
def slug(s: str) -> str:
    return (
        str(s).upper()
        .replace("/", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace("-", " ")
    )


def build_typical_week_counts(
    df: pd.DataFrame,
    *,
    complaint_col: str = "descriptor_group",   # or "descriptor"
    months: tuple[int, ...] = (6, 7, 8),
    agg: str = "median",  # "median" or "mean"
) -> pd.DataFrame:
    """
    Returns one row per (puma, complaint, dow) with typical_daily_count,
    plus a synthetic 'date' column for Kepler animation and a dow_complaint label.
    """

    out = df.copy()

    # Ensure types
    out["created_bucket"] = pd.to_datetime(out["created_bucket"], errors="coerce")
    out = out[out["created_bucket"].notna()].copy()

    # Summer-only
    out = out[out["created_bucket"].dt.month.isin(months)].copy()

    out["puma"] = out["puma"].astype(str).str.strip()
    out[complaint_col] = out[complaint_col].astype(str).str.strip()

    # Daily totals per (puma, date, complaint)
    out["date"] = out["created_bucket"].dt.normalize()
    out["dow"] = out["date"].dt.day_name()

    daily = (
        out.groupby(["puma", "date", complaint_col, "dow"], as_index=False)["complaint_count"]
           .sum()
           .rename(columns={"complaint_count": "daily_count", complaint_col: "complaint"})
    )

    # Typical-week aggregation across summer days
    if agg == "median":
        typical = (
            daily.groupby(["puma", "complaint", "dow"], as_index=False)["daily_count"]
                 .median()
                 .rename(columns={"daily_count": "typical_daily_count"})
        )
    elif agg == "mean":
        typical = (
            daily.groupby(["puma", "complaint", "dow"], as_index=False)["daily_count"]
                 .mean()
                 .rename(columns={"daily_count": "typical_daily_count"})
        )
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    # Synthetic dates for Kepler animation
    dow_to_date = {
        "Monday":    "2000-01-03",
        "Tuesday":   "2000-01-04",
        "Wednesday": "2000-01-05",
        "Thursday":  "2000-01-06",
        "Friday":    "2000-01-07",
        "Saturday":  "2000-01-08",
        "Sunday":    "2000-01-09",
    }
    typical["date"] = pd.to_datetime(typical["dow"].map(dow_to_date), errors="coerce").astype("datetime64[ns]")


    typical["dow_complaint"] = (
        typical["complaint"].map(slug)
        .astype(str)
        .replace({r"\s+": " "}, regex=True)
        .str.strip()
        .str.replace(" ", "_")
        + "__"
        + typical["dow"].astype(str)
    )

    return typical

def build_typical_week_city_relative_ratio(
    df: pd.DataFrame,
    *,
    created_col: str = "created_bucket",
    puma_col: str = "puma",
    count_col: str = "complaint_count",
    complaint_col: str = "descriptor_group",  #  preferred
    months: tuple[int, ...] = (6, 7, 8),
    agg: str = "median",               # "median" or "mean"
    synthetic_week_year: int = 2000,
) -> pd.DataFrame:
    """
    Ratio of PUMA's typical weekday complaint count
    to the citywide average for that weekday.

    Returns one row per (puma, complaint, dow).
    """

    out = df.copy()

    # choose complaint column
    if complaint_col is None:
        complaint_col = "descriptor_group" if "descriptor_group" in out.columns else "descriptor"

    # 1) datetime + summer filter
    out[created_col] = pd.to_datetime(out[created_col], errors="coerce")
    out = out[out[created_col].notna()]
    out = out[out[created_col].dt.month.isin(months)]

    # 2) daily date + weekday
    out["date"] = out[created_col].dt.normalize()
    out["dow"] = out["date"].dt.day_name()

    # 3) clean keys
    out[puma_col] = out[puma_col].astype(str).str.strip()
    out[complaint_col] = out[complaint_col].astype(str).str.strip()

    # 4) daily totals
    daily = (
        out.groupby([puma_col, "date", complaint_col, "dow"], as_index=False)[count_col]
           .sum()
           .rename(columns={
               puma_col: "puma",
               complaint_col: "complaint",
               count_col: "daily_count",
           })
    )

    # 5) typical weekday (median or mean)
    if agg == "median":
        typical = daily.groupby(["puma", "complaint", "dow"], as_index=False)["daily_count"].median()
    else:
        typical = daily.groupby(["puma", "complaint", "dow"], as_index=False)["daily_count"].mean()

    typical = typical.rename(columns={"daily_count": "typical_daily_count"})

    # 6) citywide weekday baseline
    city_baseline = (
        typical.groupby(["complaint", "dow"], as_index=False)["typical_daily_count"]
               .mean()
               .rename(columns={"typical_daily_count": "city_weekday_mean"})
    )

    out_ratio = typical.merge(city_baseline, on=["complaint", "dow"], how="left")

    out_ratio["city_relative_ratio"] = (
        out_ratio["typical_daily_count"] / out_ratio["city_weekday_mean"]
    )

    # 7) synthetic dates for Kepler animation
    dow_to_date = {
        "Monday":    f"{synthetic_week_year}-01-03",
        "Tuesday":   f"{synthetic_week_year}-01-04",
        "Wednesday": f"{synthetic_week_year}-01-05",
        "Thursday":  f"{synthetic_week_year}-01-06",
        "Friday":    f"{synthetic_week_year}-01-07",
        "Saturday":  f"{synthetic_week_year}-01-08",
        "Sunday":    f"{synthetic_week_year}-01-09",
    }
    out_ratio["date"] = pd.to_datetime(out_ratio["dow"].map(dow_to_date)).astype("datetime64[ns]")

    return out_ratio


def make_daily_table_for_model_with_puma(df, *, complaint_value=None, complaint_col="descriptor_group"):
    """
    Returns daily_df, coords
    daily_df columns: puma, dow, daily_count, puma_idx, dow_idx
    """

    x = df.copy()
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # summer only
    x = x[x["created_bucket"].dt.month.isin([6,7,8])].copy()

    if complaint_value is not None:
        x = x[x[complaint_col].astype(str) == str(complaint_value)].copy()

    x["date"] = x["created_bucket"].dt.normalize()
    x["dow"] = x["date"].dt.day_name()
    x["puma"] = x["puma"].astype(str).str.strip()

    daily_df = (
        x.groupby(["puma", "dow", "date"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={"complaint_count": "daily_count"})
    )

    # indices
    puma_labels, puma_idx = np.unique(daily_df["puma"].astype(str), return_inverse=True)

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily_df["dow"] = pd.Categorical(daily_df["dow"], categories=dow_order, ordered=True)
    daily_df = daily_df.dropna(subset=["dow"]).copy()
    dow_labels = np.array(dow_order)
    dow_idx = daily_df["dow"].cat.codes.to_numpy()

    daily_df["puma_idx"] = puma_idx[daily_df.index] if len(puma_idx) == len(daily_df) else np.unique(daily_df["puma"], return_inverse=True)[1]
    daily_df["dow_idx"] = dow_idx

    coords = {"puma": puma_labels, "dow": dow_labels}
    return daily_df, coords

def make_daily_table_for_model_with_nta(
    df,
    *,
    complaint_value=None,
    complaint_col="descriptor_group",
):
    """
    Returns:
      daily_df with columns:
        puma, nta_name, dow, date, year, daily_count,
        puma_idx, nta_idx, dow_idx, year_idx

      coords dict for PyMC (includes year)
    """

    x = df.copy()
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # Summer only
    x = x[x["created_bucket"].dt.month.isin([6, 7, 8])].copy()

    if complaint_value is not None:
        x = x[x[complaint_col].astype(str) == str(complaint_value)].copy()

    # -----------------------------
    # Normalize / derive keys
    # -----------------------------
    x["date"] = x["created_bucket"].dt.normalize()
    x["year"] = x["date"].dt.year.astype(int)
    x["dow"] = x["date"].dt.day_name()
    x["puma"] = x["puma"].astype(str).str.strip()
    x["nta_name"] = x["nta_name"].astype(str).str.strip()

    # -----------------------------
    # Aggregate to daily counts
    # -----------------------------
    daily_df = (
        x.groupby(
            ["puma", "nta_name", "dow", "date", "year"],
            as_index=False
        )["complaint_count"]
        .sum()
        .rename(columns={"complaint_count": "daily_count"})
    )

    # -----------------------------
    # Build indices
    # -----------------------------

    # PUMA
    puma_labels, puma_idx = np.unique(daily_df["puma"], return_inverse=True)

    # NTA
    nta_labels, nta_idx = np.unique(daily_df["nta_name"], return_inverse=True)

    # Day-of-week (fixed order)
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily_df["dow"] = pd.Categorical(
        daily_df["dow"],
        categories=dow_order,
        ordered=True,
    )
    daily_df = daily_df.dropna(subset=["dow"]).copy()
    dow_idx = daily_df["dow"].cat.codes.to_numpy()

    # Year (sorted, stable)
    year_labels = np.sort(daily_df["year"].unique()).astype(int)
    year_to_idx = {y: i for i, y in enumerate(year_labels)}
    year_idx = daily_df["year"].map(year_to_idx).to_numpy()

    # -----------------------------
    # Attach indices
    # -----------------------------
    daily_df["puma_idx"] = puma_idx[daily_df.index]
    daily_df["nta_idx"] = nta_idx[daily_df.index]
    daily_df["dow_idx"] = dow_idx
    daily_df["year_idx"] = year_idx

    # -----------------------------
    # Coords for PyMC
    # -----------------------------
    coords = {
        "puma": puma_labels,
        "nta": nta_labels,
        "dow": np.array(dow_order),
        "year": year_labels,
        "obs": np.arange(len(daily_df)),  # useful for mu_obs dims
    }

    return daily_df, coords


def extract_weekday(s: str) -> str:
    return str(s).split("__")[-1]

def add_typical_week_date_from_dow_complaint(
    df: pd.DataFrame,
    *,
    dow_complaint_col: str = "dow_complaint",
    weekday_col: str = "weekday",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add weekday + synthetic date columns derived from dow_complaint.

    Assumes dow_complaint format: <SOMETHING>__<Weekday>

    Returns a COPY of df with:
      - weekday (Monday..Sunday)
      - date (synthetic datetime64 for Kepler animation)
    """

    dow_to_date = {
        "Monday":    "2000-01-03",
        "Tuesday":   "2000-01-04",
        "Wednesday": "2000-01-05",
        "Thursday":  "2000-01-06",
        "Friday":    "2000-01-07",
        "Saturday":  "2000-01-08",
        "Sunday":    "2000-01-09",
    }

    out = df.copy()

    out[weekday_col] = out[dow_complaint_col].map(extract_weekday)

    out[date_col] = (
        pd.to_datetime(out[weekday_col].map(dow_to_date), errors="coerce")
        .astype("datetime64[ns]")
    )

    return out

def kepler_typical_week_from_dow_complaint(
    df: pd.DataFrame,
    *,
    puma_geojson_path: str,
    out_path: str,
    puma_col: str = "puma",
    crs: str = "EPSG:4326",
):
    """
    Export a typical-week GeoJSON for Kepler from a (puma, dow*) dataframe.

    Required columns in df:
      - puma
      - date (datetime64)  [for animation]
      - any numeric columns to visualize

    Parameters
    ----------
    df : pd.DataFrame
        Tidy dataframe (posterior or raw) with one row per (puma, dow)
    puma_geojson_path : str
        Path to PUMA GeoJSON
    out_path : str
        Output GeoJSON path
    """

    # --- Load polygons
    gdf_puma = gpd.read_file(puma_geojson_path)
    gdf_puma[puma_col] = gdf_puma["PUMA"].astype(str)

    # --- Type safety
    df = df.copy()
    df[puma_col] = df[puma_col].astype(str)
    gdf_puma[puma_col] = gdf_puma[puma_col].astype(str)

    # --- Merge
    gdf = gdf_puma.merge(df, on=puma_col, how="left")

    # --- CRS for Kepler
    gdf = gdf.to_crs(crs)

    # --- Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write GeoJSON
    gdf.to_file(out_path, driver="GeoJSON")

    print(f"✅ Kepler GeoJSON written to: {out_path}")
    return gdf

# ----------------------------
# Common table prep
# ----------------------------
def make_topn_table(
    cmp: pd.DataFrame,
    *,
    sort_by: str,
    ascending: bool = False,
    n: int = 10,
    cols: list[str] | None = None,
    label_fmt: str = "{puma} | {dow}",
) -> pd.DataFrame:
    """
    Return top-N rows of cmp sorted by sort_by, with a 'label' column for plotting.
    """
    cols = cols or ["puma", "dow", sort_by]
    df = (
        cmp.sort_values(sort_by, ascending=ascending)
           .head(n)
           .loc[:, cols]
           .copy()
    )
    df["label"] = [label_fmt.format(puma=p, dow=d) for p, d in zip(df["puma"], df["dow"])]
    # For horizontal plots, it's often nicer if the biggest is at the top:
    df = df.iloc[::-1].reset_index(drop=True)
    return df


# ----------------------------
# Plot 1: raw baseline vs posterior + interval (shrinkage-style)
# ----------------------------
def plot_topn_shrinkage_vs_raw(
    cmp: pd.DataFrame,
    *,
    sort_by: str = "city_weekday_mean",
    n: int = 10,
    raw_col: str = "city_weekday_mean",
    post_mean_col: str = "lam_mean",
    post_low_col: str = "lam_low_90",
    post_high_col: str = "lam_high_90",
    label_col: str = "nta_puma",   # <-- NEW
):
    # --- ensure we have a clean label column ---
    if label_col not in cmp.columns:
        # common case: pandas suffixes after merges
        if f"{label_col}_x" in cmp.columns or f"{label_col}_y" in cmp.columns:
            cmp = cmp.copy()
            cmp[label_col] = cmp.get(f"{label_col}_x")
            if f"{label_col}_y" in cmp.columns:
                cmp[label_col] = cmp[label_col].fillna(cmp[f"{label_col}_y"])
        else:
            raise KeyError(
                f"'{label_col}' not found in cmp. "
                f"Available label-like cols: {[c for c in cmp.columns if 'nta' in c or 'puma' in c]}"
            )

    cols = [label_col, "puma", "dow", raw_col, post_mean_col, post_low_col, post_high_col]

    # get top-N; make_topn_table will still create top["label"] using puma+dow,
    # so we'll overwrite it with an nta_puma-based label after.
    top = make_topn_table(cmp, sort_by=sort_by, ascending=False, n=n, cols=cols)

    # --- overwrite plotting label to use nta_puma for readability ---
    top["label"] = top[label_col].astype(str) + " | " + top["dow"].astype(str)

    y = np.arange(len(top))

    plt.figure(figsize=(10, 6))
    # posterior interval
    plt.hlines(
        y=y,
        xmin=top[post_low_col],
        xmax=top[post_high_col],
        alpha=0.6,
        label="Posterior 90% interval",
    )
    # posterior mean
    plt.scatter(
        top[post_mean_col],
        y,
        zorder=3,
        label="Posterior mean",
    )
    # raw baseline
    plt.scatter(
        top[raw_col],
        y,
        marker="x",
        s=80,
        zorder=4,
        label="Raw baseline",
    )

    plt.yticks(y, top["label"])
    plt.xlabel("Typical daily complaint count")
    plt.title(f"Top {n} by {sort_by}: raw baseline vs posterior (with 90% interval)")
    plt.grid(axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return top



def plot_topn_absdiff(
    cmp: pd.DataFrame,
    *,
    n: int = 10,
    raw_col: str = "city_weekday_mean",
    post_mean_col: str = "lam_mean",
    absdiff_col: str = "abs_diff",
    width_col: str = "lam_width_90",
    label_col: str = "nta_puma",   # <-- NEW
):
    # --- ensure we have a clean label column ---
    if label_col not in cmp.columns:
        # common case: pandas suffixes after merges
        if f"{label_col}_x" in cmp.columns or f"{label_col}_y" in cmp.columns:
            cmp = cmp.copy()
            cmp[label_col] = cmp.get(f"{label_col}_x")
            if f"{label_col}_y" in cmp.columns:
                cmp[label_col] = cmp[label_col].fillna(cmp[f"{label_col}_y"])
        else:
            raise KeyError(
                f"'{label_col}' not found in cmp. "
                f"Available label-like cols: {[c for c in cmp.columns if 'nta' in c or 'puma' in c]}"
            )

    cols = [label_col, "puma", "dow", raw_col, post_mean_col, absdiff_col, width_col]
    top = make_topn_table(cmp, sort_by=absdiff_col, ascending=False, n=n, cols=cols)

    # Override y-axis label to use nta_puma for readability
    top["label"] = top[label_col].astype(str) + " | " + top["dow"].astype(str)

    # Signed delta is more informative than abs_diff for a plot
    delta = top[post_mean_col] - top[raw_col]
    y = np.arange(len(top))

    plt.figure(figsize=(10, 6))
    plt.barh(
        y=y,
        width=delta,
        alpha=0.8,
        label="Posterior − raw",
    )
    plt.axvline(0, linewidth=1)
    plt.yticks(y, top["label"])
    plt.xlabel("Δ typical daily count (posterior − raw)")
    plt.title(f"Top {n} biggest absolute differences (with interval width as context)")
    plt.grid(axis="x", alpha=0.3)

    # Annotate with abs_diff and interval width (keeps it interpretable)
    for i, (d, ad, w) in enumerate(zip(delta, top[absdiff_col], top[width_col])):
        txt = f"|Δ|={ad:.2f}, w90={w:.2f}"
        # place text slightly to the right/left of bar end
        x = d + (0.02 * (abs(d) + 1)) if d >= 0 else d - (0.02 * (abs(d) + 1))
        plt.text(x, i, txt, va="center")

    plt.tight_layout()
    plt.show()

    return top



def export_idata(idata, out_path: str):
    """
    Save ArviZ InferenceData (PyMC sampling result) to netCDF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, out_path)
    print(f"✅ Saved idata -> {out_path}")
    return str(out_path)

def load_idata(path: str):
    """
    Load ArviZ InferenceData from netCDF.
    """
    idata = az.from_netcdf(path)
    print(f"✅ Loaded idata <- {path}")
    return idata

def compare_models_loo_waic(idata_model2, idata_model3, *, m2_name="Model 2", m3_name="Model 3"):
    """
    Returns:
      - loo_table: az.compare result using PSIS-LOO
      - waic_table: az.compare result using WAIC
    Requires that the idata objects include log_likelihood (PyMC usually provides this).
    """

    models = {m2_name: idata_model2, m3_name: idata_model3}

    # LOO comparison (preferred)
    loo_table = az.compare(models, ic="loo", method="stacking")
    # WAIC comparison (secondary)
    waic_table = az.compare(models, ic="waic", method="stacking")

    display(loo_table)
    display(waic_table)

    return loo_table, waic_table


def loo_diagnostics(idata, *, name="model"):
    loo = az.loo(idata, pointwise=True)
    # pareto_k is an xarray DataArray
    pk = loo.pareto_k
    frac_bad = float((pk > 0.7).mean())
    frac_very_bad = float((pk > 1.0).mean())
    print(f"[{name}] frac pareto_k > 0.7: {frac_bad:.3f} | > 1.0: {frac_very_bad:.3f}")
    return loo

def score_predictions(y_true: np.ndarray, y_pred_mean: np.ndarray):
    """
    Simple point-prediction scores from posterior predictive mean.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred_mean))
    rmse = np.sqrt(np.mean((y_true - y_pred_mean) ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}


def make_typical_week_2025(df_2025, *, complaint_col="descriptor_group", months = [6, 7, 8]):
    x = df_2025.copy()
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # summer 2025 only
    x = x[
        (x["created_bucket"].dt.year == 2025) &
        (x["created_bucket"].dt.month.isin(months))
    ].copy()

    x["date"] = x["created_bucket"].dt.normalize()
    x["dow"] = x["date"].dt.day_name()
    x["puma"] = x["puma"].astype(str).str.strip()

    daily = (
        x.groupby(["puma", "dow", "date"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={"complaint_count": "daily_count"})
    )

    typical_2025 = (
        daily.groupby(["puma", "dow"], as_index=False)["daily_count"]
             .median()
             .rename(columns={"daily_count": "observed_2025"})
    )

    return typical_2025

def make_daily_observed_2025(
    df_2025: pd.DataFrame,
    *,
    complaint_value=None,
    complaint_col="descriptor_group",
):
    x = df_2025.copy()

    # --- Clean timestamps ---
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # --- Summer 2025 filter ---
    x = x[
        (x["created_bucket"].dt.year == 2025) &
        (x["created_bucket"].dt.month.isin([6, 7, 8]))
    ].copy()

    # --- Optional complaint filter ---
    if complaint_value is not None:
        x = x[x[complaint_col].astype(str) == str(complaint_value)].copy()

    # --- Date engineering ---
    x["date"] = x["created_bucket"].dt.normalize()
    x["dow"] = x["date"].dt.day_name()

    # --- Clean geographic keys ---
    x["puma"] = x["puma"].astype(str).str.strip()
    x["nta_puma"] = x["nta_puma"].astype(str).str.strip()

    # --- Aggregate ---
    daily_obs = (
        x.groupby(["puma", "nta_puma", "date", "dow"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={"complaint_count": "daily_count"})
    )

    # Ensure datetime64[ns]
    daily_obs["date"] = pd.to_datetime(daily_obs["date"]).astype("datetime64[ns]")

    return daily_obs


def plot_coverage_curve(
    y_obs: np.ndarray,
    y_pp: np.ndarray,
    *,
    label: str,
    color: str = "C0",
):
    """
    y_obs : shape (n_obs,)
    y_pp  : shape (n_obs, n_draws)
    """

    levels = np.linspace(0.1, 0.9, 9)
    empirical = []

    for lvl in levels:
        lo = np.quantile(y_pp, (1 - lvl) / 2, axis=1)
        hi = np.quantile(y_pp, 1 - (1 - lvl) / 2, axis=1)
        covered = (y_obs >= lo) & (y_obs <= hi)
        empirical.append(covered.mean())

    plt.plot(levels, empirical, marker="o", label=label, color=color)


def score_against_real_2025_days(
    *,
    daily_2025: pd.DataFrame,
    idata,
    coords: dict,
    year_fallback: str = "nearest",   # "nearest" | "zero" | "error"
    hdi_prob: float = 0.90,
    random_seed: int = 42,
):
    df = daily_2025.copy()

    # -----------------------------
    # 1) Hygiene
    # -----------------------------
    df["puma"] = df["puma"].astype(str).str.strip()
    df["dow"]  = df["dow"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[df["date"].notna()].copy()
    df["year"] = df["date"].dt.year.astype(int)

    # -----------------------------
    # 2) Index maps
    # -----------------------------
    puma_to_idx = {str(p): i for i, p in enumerate(coords["puma"])}
    dow_to_idx  = {str(d): i for i, d in enumerate(coords["dow"])}

    df["puma_idx"] = df["puma"].map(puma_to_idx)
    df["dow_idx"]  = df["dow"].map(dow_to_idx)

    df = df.dropna(subset=["puma_idx", "dow_idx"]).copy()
    df["puma_idx"] = df["puma_idx"].astype(int)
    df["dow_idx"]  = df["dow_idx"].astype(int)

    # Year mapping if the model has year coords
    has_year = "year" in coords and coords["year"] is not None and len(coords["year"]) > 0
    if has_year:
        year_labels = [int(y) for y in coords["year"]]
        year_to_idx = {int(y): i for i, y in enumerate(year_labels)}

        def map_year(y):
            if y in year_to_idx:
                return year_to_idx[y]
            if year_fallback == "nearest":
                nearest = min(year_labels, key=lambda t: abs(t - y))
                return year_to_idx[nearest]
            if year_fallback == "zero":
                return None
            if year_fallback == "error":
                return None
            raise ValueError("year_fallback must be nearest|zero|error")

        df["year_idx"] = df["year"].map(map_year)
    else:
        df["year_idx"] = None

    # Date mapping if the model has date coords (day_shock)
    has_date = "date" in coords and coords["date"] is not None and len(coords["date"]) > 0
    if has_date:
        # Normalize coord dates to Timestamp for mapping
        coord_dates = pd.to_datetime(coords["date"]).normalize()
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(coord_dates)}
        df["date_idx"] = df["date"].map(date_to_idx)
    else:
        df["date_idx"] = None

    # -----------------------------
    # 3) Pull posterior draws
    # -----------------------------
    post = idata.posterior

    # Baseline log_lambda(puma,dow)
    log_lambda_draws = (
        post["log_lambda"]
        .stack(sample=("chain", "draw"))
        .transpose("puma", "dow", "sample")
        .values
    )  # (n_puma, n_dow, S)
    S = log_lambda_draws.shape[-1]

    # Dispersion alpha_dow(dow)
    alpha_dow_draws = (
        post["alpha_dow"]
        .stack(sample=("chain", "draw"))
        .transpose("dow", "sample")
        .values
    )  # (n_dow, S)

    # Optional year_offset(year)
    has_year_offset = "year_offset" in post.data_vars

    if has_year_offset:
        year_offset_draws = (
            post["year_offset"]
            .stack(sample=("chain", "draw"))
            .transpose("year", "sample")
            .values
        )  # (n_year, S)

    # Optional day_shock(date)
    has_day_shock = "day_shock" in post.data_vars
    if has_day_shock:
        day_shock_draws = (
            post["day_shock"]
            .stack(sample=("chain", "draw"))
            .transpose("date", "sample")
            .values
        )  # (n_date, S)

    # -----------------------------
    # 4) Build log-mu per row
    # -----------------------------
    p_idx = df["puma_idx"].to_numpy()
    d_idx = df["dow_idx"].to_numpy()

    log_mu = log_lambda_draws[p_idx, d_idx, :]  # (n_rows, S)

    # Add year offset if model has it
    if has_year_offset:
        if df["year_idx"].isna().any() or df["year_idx"].iloc[0] is None:
            if year_fallback == "zero":
                # missing year -> 0 effect
                pass
            else:
                # If nearest mapping was requested but coords missing, drop those rows
                df = df[df["year_idx"].notna()].copy()
                p_idx = df["puma_idx"].to_numpy()
                d_idx = df["dow_idx"].to_numpy()
                log_mu = log_lambda_draws[p_idx, d_idx, :]

        if len(df) > 0 and df["year_idx"].notna().all():
            y_idx = df["year_idx"].astype(int).to_numpy()
            log_mu = log_mu + year_offset_draws[y_idx, :]

    # Add day shock if model has it
    if has_day_shock:
        df = df[df["date_idx"].notna()].copy()
        p_idx = df["puma_idx"].to_numpy()
        d_idx = df["dow_idx"].to_numpy()
        log_mu = log_lambda_draws[p_idx, d_idx, :]

        # re-add year offset if applicable after filtering
        if has_year_offset and len(df) > 0 and df["year_idx"].notna().all():
            y_idx = df["year_idx"].astype(int).to_numpy()
            log_mu = log_mu + year_offset_draws[y_idx, :]

        t_idx = df["date_idx"].astype(int).to_numpy()
        log_mu = log_mu + day_shock_draws[t_idx, :]

    mu_draws = np.exp(log_mu)
    df["mu_pred_mean"] = mu_draws.mean(axis=1)

    # alpha per row (based on dow)
    alpha_draws = alpha_dow_draws[d_idx, :]  # (n_rows, S)

    # -----------------------------
    # 5) Posterior predictive y draws (NB via Gamma–Poisson)
    # -----------------------------
    rng = np.random.default_rng(random_seed)

    rate = alpha_draws / np.clip(mu_draws, 1e-9, None)
    lam_day = rng.gamma(shape=alpha_draws, scale=1.0 / rate)
    y_pp = rng.poisson(lam_day)

    lo = (1.0 - hdi_prob) / 2
    hi = 1.0 - lo
    df["y_pred_low_90"]  = np.quantile(y_pp, lo, axis=1)
    df["y_pred_high_90"] = np.quantile(y_pp, hi, axis=1)

    df["within_90_pred"] = (
        (df["daily_count"] >= df["y_pred_low_90"]) &
        (df["daily_count"] <= df["y_pred_high_90"])
    )

    df["error"] = df["daily_count"] - df["mu_pred_mean"]
    df["abs_error"] = df["error"].abs()

    summary = {
        "MAE": float(df["abs_error"].mean()),
        "Median AE": float(df["abs_error"].median()),
        "90% Coverage (predictive)": float(df["within_90_pred"].mean()),
        "N_days": int(len(df)),
        "used_year_offset": bool(has_year_offset),
        "used_day_shock": bool(has_day_shock),
    }

    return df, y_pp, summary


def crps_from_draws(y_pp, y_obs, *, pair_subsample=256, seed=42):
    """
    Approximate CRPS per row using posterior predictive draws.

    y_pp: (N, S) draws
    y_obs: (N,) observed
    pair_subsample: use K draws to approximate E|X-X'| term
    """
    rng = np.random.default_rng(seed)
    N, S = y_pp.shape
    y_obs = np.asarray(y_obs).reshape(-1,)

    # term1 = E|X - y|
    term1 = np.mean(np.abs(y_pp - y_obs[:, None]), axis=1)

    # term2 = 0.5 * E|X - X'|
    K = min(pair_subsample, S)
    idx = rng.choice(S, size=K, replace=False)
    xs = y_pp[:, idx]  # (N, K)

    # Efficient pairwise absolute differences: sort trick
    xs_sorted = np.sort(xs, axis=1)
    # E|X-X'| for sample-based distribution:
    # 2/(K^2) * sum_{i} (2i-K-1)*x_i  (for sorted x_i), then take abs already via ordering
    weights = (2*np.arange(1, K+1) - K - 1).astype(float)  # (K,)
    sum_weighted = np.sum(xs_sorted * weights[None, :], axis=1)
    e_abs_xx = (2.0 / (K*K)) * sum_weighted

    crps = term1 - 0.5 * e_abs_xx
    return crps


def elpd_from_draws(y_pp, y_obs, *, smoothing=1.0):
    """
    Estimate expected log predictive density from posterior predictive draws.

    Uses an empirical pmf from draws with additive smoothing:
      p(y) = (count(y) + smoothing) / (S + smoothing * (support_size))

    Returns: (elpd_total, elpd_mean, lpd_per_row)
    """
    y_obs = np.asarray(y_obs).astype(int)
    N, S = y_pp.shape

    lpd = np.empty(N, dtype=float)

    for i in range(N):
        draws = y_pp[i, :].astype(int)

        # empirical counts
        vals, counts = np.unique(draws, return_counts=True)
        count_map = dict(zip(vals, counts))

        # define support size as unique vals in draws plus the observed value
        support_vals = set(vals.tolist() + [int(y_obs[i])])
        K = len(support_vals)

        c = count_map.get(int(y_obs[i]), 0)
        p = (c + smoothing) / (S + smoothing * K)

        lpd[i] = np.log(p)

    elpd_total = float(np.sum(lpd))
    elpd_mean = float(np.mean(lpd))
    return elpd_total, elpd_mean, lpd

def random_summer_date(year: int) -> str:
    """
    Return a random date string (YYYY-MM-DD)
    between June 1 and August 31 of the given year.
    """

    start = date(year, 6, 1)
    end   = date(year, 8, 31)

    span_days = (end - start).days
    rand_offset = random.randint(0, span_days)

    d = start + timedelta(days=rand_offset)
    return d.isoformat()



def summarize_forecast_metrics(
    df,
    y_pp,
    *,
    level=0.90,
    crps_pair_subsample=256,
):
    y = df["daily_count"].to_numpy()
    lo = (1.0 - level) / 2
    hi = 1.0 - lo

    low = np.quantile(y_pp, lo, axis=1)
    high = np.quantile(y_pp, hi, axis=1)
    width = high - low
    coverage = np.mean((y >= low) & (y <= high))

    crps = crps_from_draws(y_pp, y, pair_subsample=crps_pair_subsample)
    elpd_total, elpd_mean, _ = elpd_from_draws(y_pp, y.astype(int), smoothing=1.0)

    mae = np.mean(np.abs(y - df["mu_pred_mean"].to_numpy())) if "mu_pred_mean" in df else np.nan

    return {
        "N": int(len(df)),
        "MAE(mu_pred_mean)": float(mae),
        f"Coverage@{int(level*100)}": float(coverage),
        f"Median PI width@{int(level*100)}": float(np.median(width)),
        "Mean CRPS": float(np.mean(crps)),
        "Median CRPS": float(np.median(crps)),
        "ELPD_total (draws)": float(elpd_total),
        "ELPD_mean (draws)": float(elpd_mean),
    }


def forest_day_puma_intervals(
    df,
    date,
    *,
    puma_col="puma",
    date_col="date",
    obs_col="daily_count",
    mean_col="mu_pred_mean",
    lo_col="y_pred_low_90",
    hi_col="y_pred_high_90",
    sort_by="abs_error",   # "abs_error", "obs", "mean", "puma"
    top_n=None,            # set e.g. 30 for readability
    title=None,
):
    """
    Forest-style plot for one date: each PUMA is a row with 90% PI + mean + observed.
    Requires df to already contain mean + interval columns for that day.
    """

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    target = pd.to_datetime(date).normalize()

    day = d.loc[d[date_col] == target].copy()
    if day.empty:
        raise ValueError(f"No rows found for {date_col} == {target.date()}")

    # Ensure puma string for labeling
    day[puma_col] = day[puma_col].astype(str).str.strip()

    # Basic checks
    needed = [puma_col, obs_col, mean_col, lo_col, hi_col]
    missing = [c for c in needed if c not in day.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute errors for sorting/annotation
    day["error"] = day[obs_col] - day[mean_col]
    day["abs_error"] = day["error"].abs()
    day["pi_width"] = day[hi_col] - day[lo_col]

    # Sort
    if sort_by == "abs_error":
        day = day.sort_values("abs_error", ascending=False)
    elif sort_by == "obs":
        day = day.sort_values(obs_col, ascending=False)
    elif sort_by == "mean":
        day = day.sort_values(mean_col, ascending=False)
    elif sort_by == "puma":
        day = day.sort_values(puma_col, ascending=True)

    # Optionally reduce rows
    if top_n is not None:
        day = day.head(int(top_n)).copy()

    # Plot positions
    y = np.arange(len(day))

    plt.figure(figsize=(10, max(4, 0.28 * len(day))))

    # Interval as horizontal line segments
    plt.hlines(
        y=y,
        xmin=day[lo_col].to_numpy(),
        xmax=day[hi_col].to_numpy(),
        linewidth=2,
        alpha=0.9,
        label="90% predictive interval",
    )

    # Predicted mean as point
    plt.scatter(
        day[mean_col].to_numpy(),
        y,
        s=35,
        marker="o",
        label="pred mean",
    )

    # Observed as x marker
    plt.scatter(
        day[obs_col].to_numpy(),
        y,
        s=35,
        marker="x",
        label="observed",
    )

    # Label y-axis with PUMA codes
    plt.yticks(y, day[puma_col].tolist())
    plt.gca().invert_yaxis()

    # Helpful title
    if title is None:
        title = f"Model intervals by PUMA on {target.date()}"

    plt.title(title)
    plt.xlabel("Daily complaints")
    plt.ylabel("PUMA")

    # Optional: vertical line at 0 isn't useful here; instead add legend & grid
    plt.grid(axis="x", alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return day  # returns the filtered/sorted table for inspection


def inspect_puma_day(df, puma, date):
    d = df.copy()
    d["puma"] = d["puma"].astype(str).str.strip()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    puma = str(puma).strip()
    date = pd.to_datetime(date).normalize()

    row = d.loc[(d["puma"] == puma) & (d["date"] == date)].copy()
    if row.empty:
        raise ValueError(f"No rows found for puma={puma} on date={date.date()}")

    # If multiple rows, keep them (could happen if you have multiple complaint types)
    # but usually it should be 1 row.
    cols = [c for c in [
        "puma","date","dow","daily_count","mu_pred_mean","y_pred_low_90","y_pred_high_90"
    ] if c in row.columns]
    print(row[cols])

    if "mu_pred_mean" in row.columns:
        row["error_mu"] = row["daily_count"] - row["mu_pred_mean"]
        row["abs_error_mu"] = row["error_mu"].abs()

    if {"y_pred_low_90","y_pred_high_90"}.issubset(row.columns):
        row["within_90_pred"] = (
            (row["daily_count"] >= row["y_pred_low_90"]) &
            (row["daily_count"] <= row["y_pred_high_90"])
        )

    return row

def plot_puma_day_interval(df, puma, date):
    r = inspect_puma_day(df, puma, date)

    # If multiple rows exist, just plot the first one
    rr = r.iloc[0]

    y = float(rr["daily_count"])
    mu = float(rr["mu_pred_mean"]) if "mu_pred_mean" in rr else np.nan
    lo = float(rr["y_pred_low_90"]) if "y_pred_low_90" in rr else np.nan
    hi = float(rr["y_pred_high_90"]) if "y_pred_high_90" in rr else np.nan

    plt.figure(figsize=(7, 1.8))
    # Interval
    if np.isfinite(lo) and np.isfinite(hi):
        plt.hlines(0, lo, hi, linewidth=6, alpha=0.7, label="90% predictive interval")
        plt.scatter([lo, hi], [0, 0], s=30)

    # Mean
    if np.isfinite(mu):
        plt.scatter([mu], [0], s=90, marker="|", label="pred mean (mu_pred_mean)")

    # Observed
    plt.scatter([y], [0], s=80, label="observed (daily_count)")

    plt.yticks([])
    plt.xlabel("complaints")
    plt.title(f"PUMA {str(puma)} on {pd.to_datetime(date).date()}  |  error={y-mu:.2f}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def normalize_summary_for_comparison(summary: dict, *, model_label: str) -> dict:
    """
    Normalize heterogeneous model summaries into a comparable schema.
    """

    out = {
        "Model": model_label,
        "N": summary.get("N"),
        "Point MAE": np.nan,
        "Point Median AE": np.nan,
        "Coverage@90": np.nan,
        "Median Interval Width@90": np.nan,
        "Mean Interval Width@90": np.nan,
        "Interval Type": None,
    }

    # Point error
    for k in summary:
        if k.startswith("MAE"):
            out["Point MAE"] = summary[k]
        if k.startswith("Median AE"):
            out["Point Median AE"] = summary[k]

    # Interval coverage
    for k in summary:
        if k.startswith("90% Coverage"):
            out["Coverage@90"] = summary[k]
            out["Interval Type"] = "predictive" if "predictive" in k else "rate"

    # Interval width
    for k in summary:
        if k.startswith("Median") and "interval width" in k:
            out["Median Interval Width@90"] = summary[k]
        if k.startswith("Mean") and "interval width" in k:
            out["Mean Interval Width@90"] = summary[k]

    return out



def rebuild_daily_cmp_2025_model3(
    *,
    daily_2025: pd.DataFrame,      # must have puma,dow,date,daily_count
    idata,                         # fitted PyMC InferenceData
    coords: dict,                  # training coords (must include puma,dow)
    df_forecast_week: pd.DataFrame | None = None,  # optional baseline
    puma_col: str = "puma",
    dow_col: str = "dow",
    date_col: str = "date",
    y_col: str = "daily_count",
    seed: int = 42,
    hdi_prob: float = 0.90,
):
    """
    Rebuild daily_cmp_2025_model3 using posterior draws:
      log_lambda[puma,dow] (required) and alpha_dow[dow] OR alpha (required).

    Returns:
      daily_cmp_2025_model3 (DataFrame),
      y_pp_model3 (ndarray: (n_rows, S))
    """

    # -----------------------------
    # 0) Copy + normalize types
    # -----------------------------
    df = daily_2025.copy()
    df[puma_col] = df[puma_col].astype(str).str.strip()
    df[dow_col] = df[dow_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    # -----------------------------
    # 1) Optional: merge baseline forecast (weekday mean)
    # -----------------------------
    if df_forecast_week is not None:
        fw = df_forecast_week.copy()
        fw[puma_col] = fw[puma_col].astype(str).str.strip()
        fw[dow_col] = fw[dow_col].astype(str)
        df = df.merge(fw, on=[puma_col, dow_col], how="left")

    # -----------------------------
    # 2) Build index maps from training coords
    # -----------------------------
    puma_to_idx = {str(p): i for i, p in enumerate(coords["puma"])}
    dow_to_idx = {str(d): i for i, d in enumerate(coords["dow"])}

    df["puma_idx"] = df[puma_col].map(puma_to_idx)
    df["dow_idx"] = df[dow_col].map(dow_to_idx)

    # Keep only rows that exist in training coords
    df = df.dropna(subset=["puma_idx", "dow_idx"]).copy()
    df["puma_idx"] = df["puma_idx"].astype(int)
    df["dow_idx"] = df["dow_idx"].astype(int)

    # -----------------------------
    # 3) Pull posterior draws
    # -----------------------------
    post = idata.posterior

    if "log_lambda" not in post:
        raise KeyError("posterior must contain 'log_lambda' with dims ('puma','dow').")

    # (puma, dow, sample)
    log_lambda_draws = (
        post["log_lambda"]
        .stack(sample=("chain", "draw"))
        .transpose("puma", "dow", "sample")
        .values
    )
    mu_grid_draws = np.exp(log_lambda_draws)  # mu = exp(log_lambda)

    # Alpha handling: alpha_dow preferred, else alpha
    if "alpha_dow" in post:
        alpha_dow = (
            post["alpha_dow"]
            .stack(sample=("chain", "draw"))
            .transpose("dow", "sample")
            .values
        )  # (dow, S)
        alpha_mode = "alpha_dow"
    elif "alpha" in post:
        alpha = post["alpha"].stack(sample=("chain", "draw")).values  # (S,)
        alpha_mode = "alpha"
    else:
        raise KeyError("posterior must contain 'alpha_dow' or 'alpha'.")

    # -----------------------------
    # 4) Select draws per row
    # -----------------------------
    p_idx = df["puma_idx"].to_numpy()
    d_idx = df["dow_idx"].to_numpy()

    mu_draws = mu_grid_draws[p_idx, d_idx, :]   # (n_rows, S)
    df["mu_pred_mean"] = mu_draws.mean(axis=1)

    if alpha_mode == "alpha_dow":
        alpha_row = alpha_dow[d_idx, :]         # (n_rows, S)
    else:
        alpha_row = alpha[None, :]              # (1, S) broadcast to (n_rows, S)

    # -----------------------------
    # 5) Posterior predictive sampling (NegBin via Gamma–Poisson mix)
    # -----------------------------
    rng = np.random.default_rng(seed)

    # rate parameter for gamma: rate = alpha / mu  (shape=alpha, scale=mu/alpha)
    rate = alpha_row / np.clip(mu_draws, 1e-9, None)
    lam_day = rng.gamma(shape=alpha_row, scale=1.0 / rate)  # (n_rows, S)
    y_pp_model3 = rng.poisson(lam_day)                      # (n_rows, S)

    # -----------------------------
    # 6) Predictive intervals + coverage
    # -----------------------------
    q_lo = (1.0 - hdi_prob) / 2.0
    q_hi = 1.0 - q_lo

    df["y_pred_low_90"]  = np.quantile(y_pp_model3, q_lo, axis=1)
    df["y_pred_high_90"] = np.quantile(y_pp_model3, q_hi, axis=1)

    y_obs = df[y_col].to_numpy()

    df["within_90_pred"] = (y_obs >= df["y_pred_low_90"]) & (y_obs <= df["y_pred_high_90"])

    # -----------------------------
    # 7) Errors
    # -----------------------------
    df["error_mu"] = y_obs - df["mu_pred_mean"]
    df["abs_error_mu"] = np.abs(df["error_mu"])

    # Optional baseline error if lam_forecast exists
    if "lam_forecast" in df.columns:
        df["error_lam_forecast"] = y_obs - df["lam_forecast"]
        df["abs_error_lam_forecast"] = np.abs(df["error_lam_forecast"])

    return df, y_pp_model3


def summarize_model_performance(df: pd.DataFrame) -> dict:
    """
    Summarize model performance.

    Supports:
    - Poisson / typical-week outputs:
        observed_2025, lam_forecast, lam_low_90, lam_high_90
    - NB daily predictive outputs:
        daily_count, mu_pred_mean, y_pred_low_90, y_pred_high_90

    Returns metrics for:
    - Point accuracy (MAE, Median AE)
    - Interval coverage
    - Interval width
    """

    # -----------------------------
    # Observed values
    # -----------------------------
    if "daily_count" in df.columns:
        y_obs = df["daily_count"].to_numpy()
    elif "observed_2025" in df.columns:
        y_obs = df["observed_2025"].to_numpy()
    else:
        raise KeyError("Need 'daily_count' or 'observed_2025'.")

    # -----------------------------
    # Point prediction
    # -----------------------------
    if "mu_pred_mean" in df.columns:
        y_hat = df["mu_pred_mean"].to_numpy()
        pred_label = "mu_pred_mean"
    elif "lam_forecast" in df.columns:
        y_hat = df["lam_forecast"].to_numpy()
        pred_label = "lam_forecast"
    else:
        raise KeyError("Need 'mu_pred_mean' or 'lam_forecast'.")

    abs_err = np.abs(y_obs - y_hat)

    # -----------------------------
    # Interval selection
    # -----------------------------
    interval_type = None

    if {"y_pred_low_90", "y_pred_high_90"}.issubset(df.columns):
        lo = df["y_pred_low_90"].to_numpy()
        hi = df["y_pred_high_90"].to_numpy()
        interval_type = "predictive"
    elif {"lam_low_90", "lam_high_90"}.issubset(df.columns):
        lo = df["lam_low_90"].to_numpy()
        hi = df["lam_high_90"].to_numpy()
        interval_type = "lam"
    else:
        lo = hi = None

    # -----------------------------
    # Coverage + width
    # -----------------------------
    if lo is not None:
        coverage = np.mean((y_obs >= lo) & (y_obs <= hi))
        width = hi - lo
        width_mean = float(np.mean(width))
        width_median = float(np.median(width))
    else:
        coverage = np.nan
        width_mean = np.nan
        width_median = np.nan

    # -----------------------------
    # Assemble results
    # -----------------------------
    out = {
        "N": int(len(df)),
        f"MAE ({pred_label})": float(np.mean(abs_err)),
        f"Median AE ({pred_label})": float(np.median(abs_err)),
    }

    if interval_type is not None:
        out[f"90% Coverage ({interval_type} interval)"] = float(coverage)
        out[f"Median {interval_type} interval width (90%)"] = width_median
        out[f"Mean {interval_type} interval width (90%)"] = width_mean

    return out