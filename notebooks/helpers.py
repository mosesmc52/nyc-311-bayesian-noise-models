import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import arviz as az
import matplotlib.pyplot as plt
from IPython.display import display

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
    Numeric value columns are rounded to the nearest .001 for cleaner visualization.
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

    # Round numeric metric columns to 3 decimals
    if value_cols:
        for col in value_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce").round(3)

    # Ensure output directory exists
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    gdf.to_file(out_path, driver="GeoJSON")

    return gdf

def export_geo_kepler(
    df: pd.DataFrame,
    *,
    geojson_path: str,
    df_key: str,
    geo_key: str,
    value_cols: list[str] | None = None,
    fill_value: float | int | None = 0,
    out_path: str | Path,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Merge an aggregated dataframe with arbitrary polygons and export a Kepler-ready GeoJSON.
    Numeric value columns are rounded to the nearest .001 for cleaner visualization.
    """

    gdf_geo = gpd.read_file(geojson_path)
    gdf_geo[df_key] = gdf_geo[geo_key].astype(str)

    df = df.copy()
    df[df_key] = df[df_key].astype(str)

    gdf = gdf_geo.merge(df, on=df_key, how="left")

    if gdf.crs is None or gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)

    if value_cols and fill_value is not None:
        for col in value_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna(fill_value)

    if value_cols:
        for col in value_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce").round(3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(out_path, driver="GeoJSON")

    return gdf


def build_typical_week_city_relative_ratio(
    df: pd.DataFrame,
    *,
    created_col: str = "created_bucket",
    puma_col: str = "puma",
    count_col: str = "complaint_count",
    complaint_col: str = "descriptor_group",
    nta_puma_col: str = "nta_puma",   # new
    months: tuple[int, ...] = (6, 7, 8),
    agg: str = "median",
    synthetic_week_year: int = 2000,
) -> pd.DataFrame:
    """
    Ratio of PUMA's typical weekday complaint count
    to the citywide average for that weekday.

    Returns one row per (puma, complaint, dow), while also carrying
    an nta_puma display label when available.
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

    if nta_puma_col in out.columns:
        out[nta_puma_col] = out[nta_puma_col].astype(str).str.strip()

    # 4) daily totals
    group_cols = [puma_col, "date", complaint_col, "dow"]
    if nta_puma_col in out.columns:
        group_cols.append(nta_puma_col)

    daily = (
        out.groupby(group_cols, as_index=False)[count_col]
           .sum()
           .rename(columns={
               puma_col: "puma",
               complaint_col: "complaint",
               count_col: "daily_count",
           })
    )

    # 5) typical weekday (median or mean)
    typical_group_cols = ["puma", "complaint", "dow"]
    if nta_puma_col in daily.columns:
        typical_group_cols.append(nta_puma_col)

    if agg == "median":
        typical = daily.groupby(typical_group_cols, as_index=False)["daily_count"].median()
    else:
        typical = daily.groupby(typical_group_cols, as_index=False)["daily_count"].mean()

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


def kepler_typical_week_from_dow_complaint(
    df: pd.DataFrame,
    *,
    puma_geojson_path: str | None = None,
    out_path: str,
    puma_col: str = "puma",
    geojson_path: str | None = None,
    df_key: str | None = None,
    geo_key: str | None = None,
    crs: str = "EPSG:4326",
):
    """
    Export a typical-week GeoJSON for Kepler from a (puma, dow*) dataframe.

    Required columns in df:
      - geography key column
      - date (datetime64)  [for animation]
      - any numeric columns to visualize

    Parameters
    ----------
    df : pd.DataFrame
        Tidy dataframe (posterior or raw) with one row per (puma, dow)
    puma_geojson_path : str
        Backward-compatible alias for geojson_path
    out_path : str
        Output GeoJSON path
    """

    geojson_path = geojson_path or puma_geojson_path
    df_key = df_key or puma_col
    geo_key = geo_key or "PUMA"

    if geojson_path is None:
        raise ValueError("Provide geojson_path (or puma_geojson_path).")

    # --- Load polygons
    gdf_geo = gpd.read_file(geojson_path)
    gdf_geo[df_key] = gdf_geo[geo_key].astype(str)

    # --- Type safety
    df = df.copy()
    df[df_key] = df[df_key].astype(str)
    gdf_geo[df_key] = gdf_geo[df_key].astype(str)

    # --- Merge
    gdf = gdf_geo.merge(df, on=df_key, how="left")

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


def plot_puma_model_vs_observed(
    df,
    dow,
    segment="top",
    n=20,
    sort_by="lam_mean",
    figsize=(10, 12)
):
    """
    Plot Bayesian model estimates vs observed complaint means.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing model and observed metrics
    dow : str
        Weekday to visualize
    segment : str
        'top', 'mid', or 'bottom'
    n : int
        Number of PUMAs to display
    sort_by : str
        Column used to rank rows ("lam_mean" or "mean_complaint_count")
    figsize : tuple
        Matplotlib figure size
    """

    df_plot = df[df["dow"] == dow].copy()

    low_col = "lam_low_90" if "lam_low_90" in df_plot.columns else "lam_mean_low_90"
    high_col = "lam_high_90" if "lam_high_90" in df_plot.columns else "lam_mean_high_90"

    if low_col not in df_plot.columns or high_col not in df_plot.columns:
        raise KeyError(
            "Expected posterior interval columns 'lam_low_90'/'lam_high_90' "
            "or 'lam_mean_low_90'/'lam_mean_high_90'."
        )

    df_plot = df_plot.sort_values(sort_by)

    total = len(df_plot)

    if segment == "top":
        df_plot = df_plot.tail(n)

    elif segment == "bottom":
        df_plot = df_plot.head(n)

    elif segment == "mid":
        mid = total // 2
        half = n // 2
        df_plot = df_plot.iloc[mid - half: mid + half]

    else:
        raise ValueError("segment must be 'top', 'mid', or 'bottom'")

    df_plot = df_plot.sort_values(sort_by)

    y = range(len(df_plot))

    fig, ax = plt.subplots(figsize=figsize)

    # credible intervals
    ax.hlines(
        y=y,
        xmin=df_plot[low_col],
        xmax=df_plot[high_col],
        color="steelblue",
        lw=2
    )

    # model mean
    ax.scatter(
        df_plot["lam_mean"],
        y,
        color="steelblue",
        s=60,
        label="Model Mean"
    )

    # observed mean
    ax.scatter(
        df_plot["mean_complaint_count"],
        y,
        marker="x",
        color="darkorange",
        s=70,
        label="Observed Mean"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["nta_puma_x"])

    ax.set_xlabel("Daily Complaint Rate")

    ax.set_title(f"Model vs Observed Noise Complaints ({dow} • {segment.upper()})")

    ax.grid(True, axis="x", alpha=0.3)

    ax.legend()

    plt.tight_layout()

    return fig, ax

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

def make_typical_week_2025(
    df_2025,
    *,
    complaint_col="descriptor_group",
    months=[6, 7, 8],
    geo_col="puma",
):
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
    x[geo_col] = x[geo_col].astype(str).str.strip()

    daily = (
        x.groupby([geo_col, "dow", "date"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={geo_col: "geo", "complaint_count": "daily_count"})
    )

    typical_2025 = (
        daily.groupby(["geo", "dow"], as_index=False)["daily_count"]
             .median()
             .rename(columns={"daily_count": "observed_2025"})
    )

    return typical_2025

def make_daily_observed_2025(
    df_2025: pd.DataFrame,
    *,
    complaint_value=None,
    complaint_col="descriptor_group",
    geo_col="puma",
    label_col="nta_puma",
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
    x[geo_col] = x[geo_col].astype(str).str.strip()
    x[label_col] = x[label_col].astype(str).str.strip()

    # --- Aggregate ---
    daily_obs = (
        x.groupby([geo_col, label_col, "date", "dow"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={geo_col: "geo", label_col: "geo_label", "complaint_count": "daily_count"})
    )

    # Ensure datetime64[ns]
    daily_obs["date"] = pd.to_datetime(daily_obs["date"]).astype("datetime64[ns]")

    return daily_obs

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




def summarize_lam_posterior(idata, value_name):
    lam_post = idata.posterior["lam"]
    geo_dim = next((dim for dim in lam_post.dims if dim in {"puma", "nta"}), None)

    if geo_dim is None:
        raise ValueError(f"Expected 'lam' to include a 'puma' or 'nta' dimension, got {lam_post.dims}.")

    mean_df = (
        lam_post
        .mean(dim=("chain", "draw"))
        .to_dataframe(name=value_name)
        .reset_index()
    )

    hdi_df = (
        az.hdi(lam_post, hdi_prob=0.90)["lam"]
        .to_dataframe(name="lam_hdi")
        .reset_index()
        .pivot_table(index=[geo_dim, "dow"], columns="hdi", values="lam_hdi")
        .reset_index()
        .rename(columns={
            "lower": f"{value_name}_low_90",
            "higher": f"{value_name}_high_90",
        })
    )

    out = mean_df.merge(hdi_df, on=[geo_dim, "dow"], how="left")
    out[f"{value_name}_width_90"] = out[f"{value_name}_high_90"] - out[f"{value_name}_low_90"]
    return out
