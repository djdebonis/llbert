from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from utils.geo_utils import parse_dms_coordinate


def _normalize_for_join(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"[^a-z0-9\-_\s]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = text.strip("-")
    return text


def clean_string_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    collapse_whitespace: bool = True,
    strip: bool = True,
    keep_numbers: bool = True,
    keep_underscore: bool = True,
    extra_chars: str = "",
) -> pd.DataFrame:
    """Clean text columns in a DataFrame while preserving a copy of the original."""
    out = df.copy()

    for col in columns:
        out[col] = out[col].astype("string")

    if lowercase:
        for col in columns:
            out[col] = out[col].str.lower()

    if remove_punctuation:
        allowed = "a-z0-9" if keep_numbers else "a-z"
        underscore = "_" if keep_underscore else ""
        pattern = rf"[^{allowed}{underscore}{re.escape(extra_chars)}\s]"
        for col in columns:
            out[col] = out[col].str.replace(pattern, "", regex=True)

    if collapse_whitespace:
        for col in columns:
            out[col] = out[col].str.replace(r"\s+", " ", regex=True)
            out[col] = out[col].str.replace(r"\s*-\s*", "-", regex=True)

    if strip:
        for col in columns:
            out[col] = out[col].str.strip()

    return out


def split_words(df: pd.DataFrame, column: str, new_column: str) -> pd.DataFrame:
    """Split a whitespace-delimited column into a Python list of tokens."""
    out = df.copy()
    out[new_column] = out[column].fillna("").astype(str).str.split(" ")
    return out


def read_sheet(path: str | Path, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read a spreadsheet from CSV or Excel into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        if sheet_name is None:
            with pd.ExcelFile(path) as xls:
                first_sheet = xls.sheet_names[0]
                return pd.read_excel(path, sheet_name=first_sheet)
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def group_and_clean_spreadsheets(
    sources: Iterable[str | Path | pd.DataFrame],
    *,
    columns: list[str] | None = None,
    group_by: list[str] | str | None = None,
    agg: Mapping[str, str] | None = None,
    **clean_kwargs,
) -> pd.DataFrame:
    """Read multiple spreadsheet files or combine in-memory DataFrames, clean them,
    and optionally aggregate along a group column.
    """
    frames = []
    for source in sources:
        if isinstance(source, pd.DataFrame):
            frames.append(source)
        else:
            frames.append(read_sheet(source))

    combined = pd.concat(frames, ignore_index=True)

    if columns:
        combined = clean_string_columns(combined, columns, **clean_kwargs)

    if group_by is not None:
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        if agg is None:
            return combined.groupby(group_cols, dropna=False, as_index=False).size().rename(columns={"size": "count"})
        return combined.groupby(group_cols, dropna=False, as_index=False).agg(agg)

    return combined


def aggregate_ll_sheets(
    data_dir: str | Path = "ll_sheets",
    coordinate_lookup: str | Path = "coordinate_dict10.xlsx",
    *,
    intersection_column: str = "intersection",
    text_column: str = "text_on_sign_exact",
    code_column: str = "code_type",
) -> pd.DataFrame:
    """Aggregate all spreadsheets in ll_sheets, clean the text, and attach coordinates from the coordinate lookup workbook using the intersection key."""
    base = Path(data_dir)
    frames = []
    for path in sorted(base.glob("*.xlsx")):
        df = read_sheet(path)
        if df.empty:
            continue
        present = list(df.columns)
        keep = [c for c in [intersection_column, text_column, code_column, "notes", "Unnamed: 3"] if c in present]
        if intersection_column not in present or text_column not in present:
            continue
        frames.append(df[keep].copy())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={text_column: "text_on_sign"})
    combined = clean_string_columns(
        combined,
        [intersection_column, "text_on_sign"],
        extra_chars="-",
    )
    combined["intersection_key"] = combined[intersection_column].map(_normalize_for_join)

    coord_df = read_sheet(coordinate_lookup)
    if "cd" in coord_df.columns:
        coord_df = coord_df.rename(columns={"cd": "coordinate_string"})
    coord_df = coord_df.copy()
    coord_df["intersection_key"] = coord_df[intersection_column].map(_normalize_for_join)
    coord_df = coord_df[["intersection_key", "coordinate_string", "city", "zip"]].dropna(subset=["intersection_key"]).drop_duplicates(subset=["intersection_key"])

    merged = combined.merge(coord_df, on="intersection_key", how="left")
    merged["latitude"] = pd.Series([np.nan] * len(merged), dtype=float)
    merged["longitude"] = pd.Series([np.nan] * len(merged), dtype=float)

    valid = merged["coordinate_string"].notna()
    if valid.any():
        parsed = merged.loc[valid, "coordinate_string"].map(parse_dms_coordinate)
        latitudes, longitudes = zip(*parsed.tolist())
        merged.loc[valid, "latitude"] = latitudes
        merged.loc[valid, "longitude"] = longitudes

    return merged
