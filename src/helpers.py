import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Pattern, Set, Union, Tuple
import re
from src.data import UFCData
from pandas.api.types import is_numeric_dtype
import logging

# Logger setup
logger = logging.getLogger(__name__)

def prepare_modeling_df(
    ufc_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    drop_r: bool = True,
    drop_b: bool = True,
    keep_last: bool = True,
    keep_career: bool = True,
    keep_ema: bool = True,
    compute_dif: bool = False,
    drop_cols: Optional[Iterable[str]] = None,
    # Fine-grained control / exceptions
    protect_all_diffs: bool = False,
    keep_diff_bases: Optional[Iterable[str]] = None,
    skip_diff_bases: Optional[Iterable[str]] = None,
    skip_diff_patterns: Optional[Iterable[str]] = None,
    keep_cols: Optional[Iterable[str]] = None,
    keep_patterns: Optional[Iterable[str]] = None,
    min_total_fights: Optional[int] = 1,
    total_fights_cols: Tuple[str, str] = ("r_total_fights", "b_total_fights"),
    as_ufcdata: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, "UFCData"]:
    """
    Prepare a UFC fights DataFrame for modeling.

    The function:
      1) Optionally filters rows by a date range.
      2) Drops user-specified columns FIRST (so no work is done on them and no *_dif are created from them).
      3) Creates numeric difference features of the form '<base>_dif = b_<base> - r_<base>'
         only for [surviving, numeric, r_/b_ pairs] and only if allowed by suffix rules or exceptions.
      4) Optionally drops the original r_* and/or b_* columns after diffs are created.
      5) Applies suffix rules (_last_n, _career, _ema) to both base columns and *_dif columns.
         You can override with explicit keeps by name/pattern or by whitelisting diff bases.
      6) Optionally wraps the result in `UFCData`.

    Suffix logic (applies to both bases and *_dif via their base name):
      - If a column's *base* matches any droppable suffix (_last_n, _career, _ema), it is dropped
        unless the corresponding keep_* flag is True OR it is explicitly kept (keep_cols / keep_patterns)
        OR (for *_dif) its base is whitelisted in keep_diff_bases OR protect_all_diffs=True.

    Parameters
    ----------
    ufc_df : pd.DataFrame
        Input dataset with columns like 'date', 'r_<feature>', 'b_<feature>', plus metadata.
    start_date : str, optional
        Inclusive lower bound for filtering by 'date'. Parsed with pandas.to_datetime.
    end_date : str, optional
        Inclusive upper bound for filtering by 'date'. Parsed with pandas.to_datetime.
    drop_r : bool, default True
        Drop all columns starting with 'r_' after computing *_dif.
    drop_b : bool, default True
        Drop all columns starting with 'b_' after computing *_dif.
    keep_last : bool, default True
        Keep features whose base ends with '_last_<n>'.
    keep_career : bool, default True
        Keep features whose base ends with '_career'.
    keep_ema : bool, default True
        Keep features whose base ends with '_ema'.
    drop_cols : Iterable[str], optional
        Exact column names to drop **first** (before any other processing).
    protect_all_diffs : bool, default False
        If True, never drop any '*_dif' columns (explicit keeps still apply).
    keep_diff_bases : Iterable[str], optional
        Base names (without '_dif') whose corresponding '*_dif' must always be kept.
        Example: ["momentum_ema"] ensures 'momentum_ema_dif' is kept.
    keep_cols : Iterable[str], optional
        Exact column names to always keep.
    keep_patterns : Iterable[str], optional
        Regex patterns; any column matching at least one pattern is always kept.
    as_ufcdata : bool, default False
        If True, return an instance of UFCData(df). Requires that UFCData is importable.

    Returns
    -------
    pd.DataFrame or UFCData
        Cleaned, modeling-ready DataFrame (or UFCData wrapper).

    Examples
    --------
    # Keep only _career, drop _ema and _last_n, but keep momentum_ema_dif as an exception:
    >>> df_ready = prepare_modeling_df(
    ...     ufc_df,
    ...     start_date="2010-01-01",
    ...     drop_r=True, drop_b=True,
    ...     keep_last=False, keep_career=True, keep_ema=False,
    ...     keep_diff_bases=["momentum_ema"]
    ... )

    # Drop specific columns first (no diffs will be computed from them):
    >>> df_ready = prepare_modeling_df(
    ...     ufc_df,
    ...     drop_cols=["r_momentum_ema", "b_momentum_ema", "r_SLpM_last_5", "b_SLpM_last_5"],
    ...     keep_last=False, keep_career=True, keep_ema=False
    ... )

    Notes
    -----
    - Difference direction is always Blue minus Red (b_ - r_).
    - Non-numeric r_/b_ pairs (e.g., stance) are ignored for diffs.
    """
    # ---- Normalize inputs
    drop_cols_list: List[str] = list(drop_cols or [])
    keep_diff_bases_set: Set[str] = set(keep_diff_bases or [])
    keep_cols_set: Set[str] = set(keep_cols or [])
    pattern_objs: List[Pattern[str]] = [re.compile(p) for p in (keep_patterns or [])]

    skip_diff_bases_set: Set[str] = set(skip_diff_bases or [])
    skip_diff_pattern_objs: List[re.Pattern] = [re.compile(p) for p in (skip_diff_patterns or [])]
    df = ufc_df.copy()

    # ---- 1) Filter by date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if start_date is not None:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.to_datetime(end_date)]

    # ---- 1b) Filter by minimum total fights
    if min_total_fights is not None:
        r_fights_col, b_fights_col = total_fights_cols
        if r_fights_col in df.columns and b_fights_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[r_fights_col]):
                df[r_fights_col] = pd.to_numeric(df[r_fights_col], errors="coerce")
            if not pd.api.types.is_numeric_dtype(df[b_fights_col]):
                df[b_fights_col] = pd.to_numeric(df[b_fights_col], errors="coerce")
            before_size = len(df)
            df = df[(df[r_fights_col] >= min_total_fights) & (df[b_fights_col] >= min_total_fights)]
            if verbose:
                print(f"[INFO] Filtered by min_total_fights={min_total_fights}: {before_size} → {len(df)} rows")
        else:
            if verbose:
                missing = [c for c in (r_fights_col, b_fights_col) if c not in df.columns]
                print(f"[WARN] Skipping min_total_fights filter; missing columns: {missing}")

    # ---- 2) Drop user-specified columns FIRST (but NEVER drop those in keep_cols/keep_patterns)
    if drop_cols_list:
        to_drop = [
            c for c in drop_cols_list
            if c in df.columns
            and c not in keep_cols_set
            and not any(p.search(c) for p in pattern_objs)
        ]
        if to_drop:
            df = df.drop(columns=to_drop, errors="ignore")

    # ---- Build suffix rules
    suffix_any = re.compile(r"_last_\d+|_career|_ema")
    keep_parts: List[str] = []
    if keep_last:
        keep_parts.append(r"_last_\d+")
    if keep_career:
        keep_parts.append(r"_career")
    if keep_ema:
        keep_parts.append(r"_ema")
    keep_pattern: Optional[Pattern[str]] = re.compile("(" + "|".join(keep_parts) + ")") if keep_parts else None

    # ---- Helpers
    def base_from_col(col: str) -> str:
        return col[:-4] if col.endswith("_dif") else col

    def allowed_by_suffix(base: str) -> bool:
        if not suffix_any.search(base):
            return True
        if keep_pattern is None:
            return False
        return bool(keep_pattern.search(base))

    def explicit_keep(col: str) -> bool:
        if col in keep_cols_set:
            return True
        return any(p.search(col) for p in pattern_objs)

    def explicit_skip_diff(col_dif: str, base: str) -> bool:
        if base in skip_diff_bases_set:
            return True
        if any(p.search(base) or p.search(col_dif) for p in skip_diff_pattern_objs):
            return True
        return False

    # ---- 3) Create *_dif only when allowed
    if compute_dif:
        r_cols = [c for c in df.columns if c.startswith("r_")]
        b_cols = [c for c in df.columns if c.startswith("b_")]
        common_feats = sorted(set(c[2:] for c in r_cols) & set(c[2:] for c in b_cols))
    
        for feat in common_feats:
            col_r = f"r_{feat}"
            col_b = f"b_{feat}"
            if col_r in df.columns and col_b in df.columns:
                if is_numeric_dtype(df[col_r]) and is_numeric_dtype(df[col_b]):
                    base = feat
                    col_dif = f"{feat}_dif"
    
                    if explicit_skip_diff(col_dif, base):
                        continue
    
                    if (
                        explicit_keep(col_dif)
                        or protect_all_diffs
                        or base in keep_diff_bases_set
                        or allowed_by_suffix(base)
                    ):
                        df[col_dif] = df[col_b] - df[col_r]

    # ---- 4) Optionally drop base r_/b_ columns,
    #         but NEVER drop those protected by keep_cols/keep_patterns
    if drop_r:
        df = df.drop(
            columns=[
                c for c in df.columns
                if c.startswith("r_")
                and c not in keep_cols_set
                and not any(p.search(c) for p in pattern_objs)
            ],
            errors="ignore"
        )
    if drop_b:
        df = df.drop(
            columns=[
                c for c in df.columns
                if c.startswith("b_")
                and c not in keep_cols_set
                and not any(p.search(c) for p in pattern_objs)
            ],
            errors="ignore"
        )

    # ---- 5) Suffix filtering on everything (explicit keeps still win)
    def should_drop(col: str) -> bool:
        if explicit_keep(col):
            return False
        if col.endswith("_dif") and protect_all_diffs:
            return False
        base = base_from_col(col)
        if col.endswith("_dif"):
            if base in keep_diff_bases_set:
                return False
            if explicit_skip_diff(col, base):
                return True
        if not allowed_by_suffix(base):
            return True
        return False

    cols_to_drop = [c for c in df.columns if should_drop(c)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # ---- 6) Optional wrapper
    if as_ufcdata:
        if UFCData is None:
            raise RuntimeError("UFCData is not available for as_ufcdata=True. Adjust the import.")
        return UFCData(df)

    return df

def build_history_features(
    df: pd.DataFrame,
    metrics: Iterable[str],
    corners: Tuple[str, str] = ("r", "b"),
    last_windows: Iterable[int] = (5,),   # e.g., (3, 5) to emit both last_3 and last_5
    alpha: float = 0.40,                  # fixed alpha for EMA
    seed_ema_with_first: bool = True,     # kept for API compatibility; EMA is always initialized with first value
    min_samples_lastn: int = 1,           # 1: use whatever is available if < n; 0: NaN if < n
    date_col: str = "date",
    name_cols: Tuple[str, str] = ("r_name", "b_name"),
    sort_by_date: bool = True,
    inplace: bool = False,
    compute_last: bool = True,
    compute_career: bool = True,
    compute_ema: bool = True,
    drop_raw_metrics: bool = False,    
) -> pd.DataFrame:
    """
    Adds leakage-safe historical feature columns for each metric and corner:
      - {corner}_{m}_last_{n}   (if compute_last=True)
      - {corner}_{m}_career     (if compute_career=True)
      - {corner}_{m}_ema        (if compute_ema=True; emits EMA from prior history)

    Leakage safety:
      - For each row t, features are computed using ONLY prior fights (< t) for the same fighter.
      - State (history and EMA) is updated AFTER features are written for row t.

    Notes on EMA:
      - EMA state is always initialized with the first observed value for a fighter/metric.
      - The first EMA *output* remains NaN because it represents the 'previous EMA'.

    Ordering:
      - Rows are stably sorted by `date_col` and (if present) by `event_id`, `bout_order`,
        with original row order as the final tie-breaker (stable mergesort).

    Raw metric dropping:
      - If `drop_raw_metrics=True` and at least one of {compute_last, compute_career, compute_ema} is True,
        raw per-fight metric columns {corner}_{m} used to build history features are dropped to avoid leakage in modeling.
    """

    if not inplace:
        df = df.copy()

    # Defensive: normalize inputs
    last_windows = tuple(last_windows) if isinstance(last_windows, Iterable) else (5,)

    # Stable chronological ordering (with dynamic tie-breakers)
    if sort_by_date:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["_orig_idx___"] = np.arange(len(df))  # ultimate stable tie-breaker

        sort_keys = [date_col]
        for k in ("event_id", "bout_order"):
            if k in df.columns:
                sort_keys.append(k)
        sort_keys.append("_orig_idx___")

        df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    r_name_col, b_name_col = name_cols

    # Detect metrics that exist for both corners
    existing_metric_cols: List[str] = []
    for m in metrics:
        if all(f"{c}_{m}" in df.columns for c in corners):
            existing_metric_cols.append(m)

    # Pre-create output columns
    for m in existing_metric_cols:
        if compute_last:
            for n in last_windows:
                for c in corners:
                    col = f"{c}_{m}_last_{n}"
                    if col not in df.columns:
                        df[col] = np.nan
        if compute_career:
            for c in corners:
                col = f"{c}_{m}_career"
                if col not in df.columns:
                    df[col] = np.nan
        if compute_ema:
            for c in corners:
                col = f"{c}_{m}_ema"
                if col not in df.columns:
                    df[col] = np.nan

    # State structures
    history: Dict[str, Dict[str, List[float]]] = {}  # history[fighter][m] = list of past values (< t)
    ema_state: Dict[str, Dict[str, float]] = {}      # ema_state[fighter][m] = previous EMA value

    def ensure_fighter(f: str):
        if f not in history:
            history[f] = {m: [] for m in existing_metric_cols}
        if f not in ema_state:
            ema_state[f] = {m: None for m in existing_metric_cols}

    # Leakage-safe chronological loop
    for idx, row in df.iterrows():
        # (a) compute features using prior history only
        for c, name_col in zip(corners, name_cols):
            fighter = row[name_col]
            ensure_fighter(fighter)

            for m in existing_metric_cols:
                past_vals = history[fighter][m]

                # last_n
                if compute_last:
                    for n in last_windows:
                        if len(past_vals) == 0:
                            val = np.nan
                        else:
                            if len(past_vals) < n and min_samples_lastn == 0:
                                val = np.nan
                            else:
                                take = past_vals[-n:] if len(past_vals) >= n else past_vals
                                val = float(np.nanmean(take)) if len(take) > 0 else np.nan
                        df.at[idx, f"{c}_{m}_last_{n}"] = val

                # career
                if compute_career:
                    career_val = float(np.nanmean(past_vals)) if len(past_vals) > 0 else np.nan
                    df.at[idx, f"{c}_{m}_career"] = career_val

                # ema (emit previous EMA)
                if compute_ema:
                    prev_ema = ema_state[fighter][m]
                    df.at[idx, f"{c}_{m}_ema"] = float(prev_ema) if prev_ema is not None else np.nan

        # (b) update history and EMA with the current fight (post-write)
        for c, name_col in zip(corners, name_cols):
            fighter = row[name_col]
            ensure_fighter(fighter)

            for m in existing_metric_cols:
                raw_col = f"{c}_{m}"
                value = row[raw_col]

                if pd.notna(value):
                    # Update history with current observed value
                    val = float(value)
                    history[fighter][m].append(val)

                    if compute_ema:
                        prev_ema = ema_state[fighter][m]
                        if prev_ema is None:
                            # Always initialize EMA state with the first observed value.
                            ema_state[fighter][m] = val
                        else:
                            ema_state[fighter][m] = float(alpha * val + (1.0 - alpha) * prev_ema)

    # Optionally drop raw per-fight metric columns to avoid leakage in modeling
    if drop_raw_metrics and (compute_last or compute_career or compute_ema):
        raw_cols_to_drop = []
        for m in existing_metric_cols:
            for c in corners:
                col = f"{c}_{m}"
                if col in df.columns:
                    raw_cols_to_drop.append(col)
        if raw_cols_to_drop:
            df = df.drop(columns=raw_cols_to_drop)

    # Cleanup helper column if created
    if "_orig_idx___" in df.columns:
        df = df.drop(columns=["_orig_idx___"])

    return df

# ---- Helpers to visualize history features ----

def list_history_columns(
    metrics: Iterable[str],
    last_windows: Iterable[int] = (3, 5),
    include_last: bool = True,
    include_career: bool = True,
    include_ema: bool = True,
    corners: Tuple[str, str] = ("r", "b"),
) -> List[str]:
    """
    Build the list of expected history feature columns for the given metrics.
    """
    cols = []
    for m in metrics:
        if include_last:
            for n in last_windows:
                for c in corners:
                    cols.append(f"{c}_{m}_last_{n}")
        if include_career:
            for c in corners:
                cols.append(f"{c}_{m}_career")
        if include_ema:
            for c in corners:
                cols.append(f"{c}_{m}_ema")
    return cols


def view_history_features(
    df: pd.DataFrame,
    metrics: Iterable[str],
    last_windows: Iterable[int] = (3, 5),
    include_last: bool = True,
    include_career: bool = True,
    include_ema: bool = True,
    corners: Tuple[str, str] = ("r", "b"),
    extra_id_cols: Iterable[str] = ("date", "r_name", "b_name", "division"),
    tail: int = 20,
) -> pd.DataFrame:
    """
    Return a dataframe slice with identifier columns + the history features for the given metrics.
    """
    hist_cols = list_history_columns(
        metrics=metrics,
        last_windows=last_windows,
        include_last=include_last,
        include_career=include_career,
        include_ema=include_ema,
        corners=corners,
    )
    cols = [c for c in extra_id_cols if c in df.columns] + [c for c in hist_cols if c in df.columns]
    return df[cols].tail(tail)


def view_fighter_history(
    df: pd.DataFrame,
    fighter_name: str,
    metrics: Iterable[str],
    last_windows: Iterable[int] = (3, 5),
    include_last: bool = True,
    include_career: bool = True,
    include_ema: bool = True,
    corners: Tuple[str, str] = ("r", "b"),
    extra_id_cols: Iterable[str] = ("date", "r_name", "b_name", "division", "method"),
    head: int = 15,
) -> pd.DataFrame:
    """
    Filter fights where the fighter appears (red or blue) and show their history features.
    """
    mask = (df.get("r_name", "") == fighter_name) | (df.get("b_name", "") == fighter_name)
    sub = df.loc[mask].copy()

    hist_cols = list_history_columns(
        metrics=metrics,
        last_windows=last_windows,
        include_last=include_last,
        include_career=include_career,
        include_ema=include_ema,
        corners=corners,
    )
    cols = [c for c in extra_id_cols if c in sub.columns] + [c for c in hist_cols if c in sub.columns]
    return sub[cols].head(head)