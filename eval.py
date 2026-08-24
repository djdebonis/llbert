import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from data_utils import split_indices  # noqa: E402
from train import (  # noqa: E402
    CoordinateRegressor,
    haversine_km,
    move_features_to_device,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sign coordinate model.")
    parser.add_argument("--data-file", default="training.csv")
    parser.add_argument("--model-path", default="output")
    parser.add_argument("--output-file", default="predictions.csv")
    parser.add_argument("--plot-file", default="plots/prediction_map.png")
    parser.add_argument(
        "--scatter-plot-file",
        default="plots/predicted_vs_actual.png",
        help="2x1 scatter plot of predicted vs actual latitude and longitude.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device to use for evaluation. Defaults to auto-select `cuda` if "
            "available else `cpu`. `cuda` is optional."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=1992,
        help="Must match the seed used during training for a consistent split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Must match the test-size used during training.",
    )
    return parser.parse_args()


def get_device(requested_device):
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("WARNING: cuda requested but not available, falling back to cpu")
            return torch.device("cpu")
        return device
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_head(model_path, filename, encoder, config, device):
    head = CoordinateRegressor(
        embedding_dim=encoder.get_embedding_dimension(),
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    head.load_state_dict(
        torch.load(
            os.path.join(model_path, filename),
            map_location=device,
        )
    )
    head.eval()
    return head


def load_model(model_path, device):
    with open(os.path.join(model_path, "coordinate_config.json")) as f:
        config = json.load(f)

    trained_encoder = SentenceTransformer(model_path, device=str(device))
    trained_encoder.to(device)
    trained_head = load_head(
        model_path, "coordinate_head.pt", trained_encoder, config, device
    )
    trained_encoder.eval()

    initial_encoder_path = os.path.join(model_path, "initial_encoder")
    if os.path.exists(initial_encoder_path):
        base_encoder = SentenceTransformer(initial_encoder_path, device=str(device))
    else:
        base_encoder = SentenceTransformer(config["model_name"], device=str(device))
    base_encoder.to(device)
    initial_head_path = os.path.join(model_path, "initial_coordinate_head.pt")
    initial_head = None
    if os.path.exists(initial_head_path):
        initial_head = load_head(
            model_path, "initial_coordinate_head.pt", base_encoder, config, device
        )
        base_encoder.eval()

    return (
        base_encoder,
        initial_head,
        trained_encoder,
        trained_head,
        np.array(config["coord_mean"]),
        np.array(config["coord_std"]),
    )


@torch.no_grad()
def predict(encoder, head, texts, coord_mean, coord_std, device, batch_size):
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        features = encoder.preprocess(batch)
        features = move_features_to_device(features, device)
        embeddings = encoder(features)["sentence_embedding"]
        batch_predictions = head(embeddings).cpu().numpy()
        predictions.append(batch_predictions)

    normalized = np.vstack(predictions)
    return normalized * coord_std + coord_mean


def make_prediction_plot(results, plot_file):
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in results.iterrows():
        ax.plot(
            [row["longitude"], row["predicted_longitude"]],
            [row["latitude"], row["predicted_latitude"]],
            color="0.75",
            linewidth=0.4,
            alpha=0.35,
            zorder=1,
        )

    ax.scatter(
        results["predicted_longitude"],
        results["predicted_latitude"],
        s=12,
        color="#d62728",
        alpha=0.45,
        label="predicted",
        zorder=2,
    )
    ax.scatter(
        results["longitude"],
        results["latitude"],
        s=18,
        color="#1f77b4",
        alpha=0.75,
        label="actual",
        zorder=3,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Sign Text Coordinate Predictions")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)


def make_predicted_vs_actual_plot(results, plot_file):
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 9))

    plot_specs = [
        ("latitude", "initial_predicted_latitude", "predicted_latitude", "Latitude"),
        (
            "longitude",
            "initial_predicted_longitude",
            "predicted_longitude",
            "Longitude",
        ),
    ]
    for ax, (actual_col, initial_col, predicted_col, title) in zip(axes, plot_specs):
        actual = results[actual_col]
        initial = results[initial_col] if initial_col in results else None
        predicted = results[predicted_col]
        values = [actual, predicted]
        if initial is not None:
            values.append(initial)
        lower = min(series.min() for series in values)
        upper = max(series.max() for series in values)

        if initial is not None:
            ax.scatter(
                actual,
                initial,
                s=14,
                alpha=0.45,
                color="#d62728",
                label="before training",
            )
        ax.scatter(
            actual,
            predicted,
            s=14,
            alpha=0.55,
            color="#1f77b4",
            label="after training",
        )
        ax.plot([lower, upper], [lower, upper], color="0.25", linewidth=1)
        ax.set_xlabel(f"actual {actual_col}")
        ax.set_ylabel(f"predicted {actual_col}")
        ax.set_title(title)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device(args.device)
    (
        base_encoder,
        initial_head,
        trained_encoder,
        trained_head,
        coord_mean,
        coord_std,
    ) = load_model(args.model_path, device)

    data = pd.read_csv(args.data_file).dropna(
        subset=["text", "latitude", "longitude"]
    )
    data = data.reset_index(drop=True)
    texts = data["text"].astype(str).tolist()
    predicted = predict(
        trained_encoder,
        trained_head,
        texts,
        coord_mean,
        coord_std,
        device,
        args.batch_size,
    )
    true_coords = data[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    errors = haversine_km(predicted, true_coords)

    results = data.copy()
    results["predicted_latitude"] = predicted[:, 0]
    results["predicted_longitude"] = predicted[:, 1]
    if initial_head is not None:
        initial_predicted = predict(
            base_encoder,
            initial_head,
            texts,
            coord_mean,
            coord_std,
            device,
            args.batch_size,
        )
        results["initial_predicted_latitude"] = initial_predicted[:, 0]
        results["initial_predicted_longitude"] = initial_predicted[:, 1]
    results["error_km"] = errors

    # Reproduce the same train/val split used during training.
    train_indices, val_indices = split_indices(
        data, test_size=args.test_size, seed=args.seed
    )
    val_mask = np.zeros(len(results), dtype=bool)
    val_mask[val_indices] = True
    results["split"] = np.where(val_mask, "val", "train")

    results.to_csv(args.output_file, index=False)
    make_prediction_plot(results, args.plot_file)
    make_predicted_vs_actual_plot(results, args.scatter_plot_file)

    print(f"Wrote predictions to {args.output_file}")
    print(f"Wrote plot to {args.plot_file}")
    print(f"Wrote scatter plot to {args.scatter_plot_file}")

    def _fmt(errs):
        return (
            f"mean={np.mean(errs):.3f}  "
            f"median={np.median(errs):.3f}  "
            f"p90={np.percentile(errs, 90):.3f}"
        )

    def _intersection_agg_errors(df):
        """Average per-bag predictions to one coordinate per intersection."""
        agg = (
            df.groupby("intersection")
            .agg(
                pred_lat=("predicted_latitude", "mean"),
                pred_lon=("predicted_longitude", "mean"),
                true_lat=("latitude", "first"),
                true_lon=("longitude", "first"),
            )
            .reset_index()
        )
        return haversine_km(
            agg[["pred_lat", "pred_lon"]].to_numpy(dtype=np.float32),
            agg[["true_lat", "true_lon"]].to_numpy(dtype=np.float32),
        )

    val_df = results[results["split"] == "val"]
    train_df = results[results["split"] == "train"]

    val_bag_errors = val_df["error_km"].to_numpy()
    train_bag_errors = train_df["error_km"].to_numpy()
    val_agg_errors = _intersection_agg_errors(val_df)
    train_agg_errors = _intersection_agg_errors(train_df)

    n_val = val_df["intersection"].nunique() if "intersection" in val_df.columns else "?"
    n_train = train_df["intersection"].nunique() if "intersection" in train_df.columns else "?"

    print()
    print(f"[all bags  ({len(errors):>6} rows)]  {_fmt(errors)}")
    print(f"[train bags({len(train_bag_errors):>6} rows)]  {_fmt(train_bag_errors)}")
    print(f"[val bags  ({len(val_bag_errors):>6} rows)]  {_fmt(val_bag_errors)}")
    print(f"[train agg ({n_train:>3} intersections)]  {_fmt(train_agg_errors)}")
    print(f"[val agg   ({n_val:>3} intersections)]  {_fmt(val_agg_errors)}  ← generalization")


if __name__ == "__main__":
    main()
