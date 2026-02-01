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
    df["month_year"] = df["month"].astype("string") + "__" + df["created_bucket"].dt.year.astype("Int64").astype("string")



    df["descriptor_group"] = df["descriptor"].map(descriptor_group).astype("string")

    # --- build dow_complaint from aggregated descriptor_group (NOT raw descriptor) ---
    df["dow_complaint"] = (
        df["descriptor_group"]
        .astype("string")
        .str.upper()
        .str.replace(r"[,_/]", " ", regex=True)  # keep it safe for tokenizing
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
):
    cols = ["puma", "dow", raw_col, post_mean_col, post_low_col, post_high_col]
    top = make_topn_table(cmp, sort_by=sort_by, ascending=False, n=n, cols=cols)

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
    width_col: str = "lam_width_90"
):
    cols = ["puma", "dow", raw_col, post_mean_col, absdiff_col, width_col]
    top = make_topn_table(cmp, sort_by=absdiff_col, ascending=False, n=n, cols=cols)

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


def make_typical_week_2025(df_2025, *, complaint_col="descriptor_group"):
    x = df_2025.copy()
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # summer 2025 only
    x = x[
        (x["created_bucket"].dt.year == 2025) &
        (x["created_bucket"].dt.month.isin([6, 7, 8]))
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
    x["created_bucket"] = pd.to_datetime(x["created_bucket"], errors="coerce")
    x = x[x["created_bucket"].notna()].copy()

    # Summer 2025 filter
    x = x[
        (x["created_bucket"].dt.year == 2025) &
        (x["created_bucket"].dt.month.isin([6, 7, 8]))
    ].copy()

    if complaint_value is not None:
        x = x[x[complaint_col].astype(str) == str(complaint_value)].copy()

    x["date"] = x["created_bucket"].dt.normalize()
    x["dow"] = x["date"].dt.day_name()
    x["puma"] = x["puma"].astype(str).str.strip()

    daily_obs = (
        x.groupby(["puma", "date", "dow"], as_index=False)["complaint_count"]
         .sum()
         .rename(columns={"complaint_count": "daily_count"})
    )

    # ensure datetime64[ns]
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
