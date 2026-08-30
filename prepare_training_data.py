import argparse
import random
from pathlib import Path

import pandas as pd

from utils.cleaning import aggregate_ll_sheets

BASE_COLUMNS = ["intersection", "text_on_sign_exact", "latitude", "longitude"]
OPTIONAL_COLUMNS = ["code_type"]
EXCLUDED_INTERSECTIONS = {"56th-pena"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create bootstrapped sign-text bags for coordinate training."
    )
    parser.add_argument(
        "-i",
        "--input-file",
        default="training_data_raw.csv",
        help="Raw pandas-exported CSV.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="training.csv",
        help="Prepared training CSV.",
    )
    parser.add_argument(
        "--from-ll-sheets",
        action="store_true",
        help="Aggregate ll_sheets/*.xlsx and coordinate_dict10.xlsx into a raw training CSV.",
    )
    parser.add_argument(
        "--sheet-dir",
        default="ll_sheets",
        help="Directory containing spreadsheet exports to aggregate.",
    )
    parser.add_argument(
        "--coord-file",
        default="coordinate_dict10.xlsx",
        help="Workbook containing the intersection-to-coordinate lookup.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1992,
        help="Random seed for deterministic bootstrapping.",
    )
    parser.add_argument(
        "--bag-size",
        type=int,
        default=8,
        help=(
            "Total number of sign texts per training row. Street signs fill "
            "slots first; remaining slots are randomly sampled from all signs."
        ),
    )
    parser.add_argument(
        "--samples-per-intersection",
        type=int,
        default=100,
        help="Bootstrap samples to create for each intersection.",
    )
    parser.add_argument(
        "--separator",
        default=" | ",
        help="Separator used when joining sampled sign texts.",
    )
    return parser.parse_args()


def load_raw_data(path):
    data = pd.read_csv(path)
    missing = sorted(set(BASE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = BASE_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in data.columns]
    data = data[keep].copy()
    data = data.dropna(subset=BASE_COLUMNS)
    data["text_on_sign_exact"] = data["text_on_sign_exact"].astype(str).str.strip()
    data = data[data["text_on_sign_exact"] != ""]
    data = data[~data["intersection"].isin(EXCLUDED_INTERSECTIONS)]
    return data


def build_raw_data_from_sheets(sheet_dir, coord_file, raw_output_path):
    raw_data = aggregate_ll_sheets(sheet_dir, coord_file)
    if raw_data.empty:
        raise ValueError(f"No data found in {sheet_dir} / {coord_file}.")

    raw_data = raw_data.rename(columns={"text_on_sign": "text_on_sign_exact"})
    raw_data = raw_data[["intersection", "text_on_sign_exact", "latitude", "longitude"] + [c for c in ["code_type"] if c in raw_data.columns]].copy()
    raw_data = raw_data.dropna(subset=["intersection", "text_on_sign_exact", "latitude", "longitude"])
    raw_data = raw_data[raw_data["text_on_sign_exact"].astype(str).str.strip() != ""]
    raw_data = raw_data[~raw_data["intersection"].isin(EXCLUDED_INTERSECTIONS)]

    raw_data.to_csv(raw_output_path, index=False)
    print(f"Wrote raw sheet aggregation to {raw_output_path}")
    return raw_data


def make_bootstrap_data(data, bag_size, samples_per_intersection, seed, separator):
    if bag_size < 1:
        raise ValueError("--bag-size must be at least 1")
    if samples_per_intersection < 1:
        raise ValueError("--samples-per-intersection must be at least 1")

    has_code_type = "code_type" in data.columns
    rows = []
    grouped = data.groupby("intersection", sort=True)
    for intersection, group in grouped:
        latitude = group["latitude"].mean()
        longitude = group["longitude"].mean()
        all_texts = group["text_on_sign_exact"].tolist()

        # Street signs are the most geographically discriminative; always
        # include them first up to bag_size, then fill remaining slots randomly.
        if has_code_type:
            street_texts = group[group["code_type"] == "street_sign"][
                "text_on_sign_exact"
            ].tolist()
        else:
            street_texts = []

        for sample_id in range(samples_per_intersection):
            rng_state = seed + len(rows)

            guaranteed = street_texts[:bag_size]
            remaining = bag_size - len(guaranteed)
            if remaining > 0:
                filler = group.sample(
                    n=remaining, replace=True, random_state=rng_state
                )["text_on_sign_exact"].tolist()
            else:
                filler = []

            bag = guaranteed + filler
            # Shuffle so position in the sequence doesn't encode sign type.
            random.Random(rng_state).shuffle(bag)

            rows.append(
                {
                    "intersection": intersection,
                    "sample_id": sample_id,
                    "text": separator.join(bag),
                    "latitude": latitude,
                    "longitude": longitude,
                    "unique_sign_count": len(set(all_texts)),
                    "raw_sign_count": len(all_texts),
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.from_ll_sheets:
        raw_path = Path(args.input_file)
        raw_data = build_raw_data_from_sheets(
            args.sheet_dir,
            args.coord_file,
            raw_path,
        )
    else:
        raw_data = load_raw_data(args.input_file)

    training_data = make_bootstrap_data(
        raw_data,
        bag_size=args.bag_size,
        samples_per_intersection=args.samples_per_intersection,
        seed=args.seed,
        separator=args.separator,
    )
    training_data.to_csv(args.output_file, index=False)
    print(
        f"Wrote {len(training_data):,} rows from "
        f"{raw_data['intersection'].nunique():,} intersections to {args.output_file}"
    )


if __name__ == "__main__":
    main()
