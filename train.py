import argparse
import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import LoggingHandler, SentenceTransformer
from data_utils import split_indices
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class SignCoordinateDataset(Dataset):
    def __init__(self, texts, coordinates):
        self.texts = list(texts)
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.coordinates[index]


class EmbeddingCoordinateDataset(Dataset):
    def __init__(self, embeddings, coordinates):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return self.embeddings[index], self.coordinates[index]


class CoordinateRegressor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, embeddings):
        return self.layers(embeddings)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sign text to lat/lon model.")
    parser.add_argument("--data-file", default="training.csv")
    parser.add_argument("--output-path", default="output")
    parser.add_argument(
        "--model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device to use for training. Defaults to auto-select `cuda` if available "
            "else `cpu`. `cuda` is optional and will fall back to `cpu` when unavailable."
        ),
    )
    parser.add_argument("--seed", type=int, default=1992)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers to prefetch batches while the GPU trains.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,

        default=5,
        help=(
            "Write the best checkpoint to disk every N epochs (and at the final "
            "epoch). Validation still runs every epoch."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Train only the coordinate head; keep the sentence encoder fixed.",
    )
    parser.add_argument(
        "--freeze-transformer-layers",
        type=int,
        default=0,
        help="Freeze the first N transformer layers in the sentence encoder.",
    )
    parser.add_argument(
        "--freeze-attention",
        action="store_true",
        help=(
            "Freeze self-attention parameters while leaving other encoder "
            "params trainable."
        ),
    )
    return parser.parse_args()


def get_device(requested_device):
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_coordinates(coordinates):
    mean = coordinates.mean(axis=0)
    std = coordinates.std(axis=0)
    std[std == 0] = 1.0
    return (coordinates - mean) / std, mean, std


def move_features_to_device(features, device, non_blocking=False):
    return {
        key: (
            value.to(device, non_blocking=non_blocking)
            if torch.is_tensor(value)
            else value
        )
        for key, value in features.items()
    }


def move_batch_to_device(features, labels, device, pin_memory=False):
    non_blocking = pin_memory and device.type == "cuda"
    if pin_memory and device.type == "cuda":
        features = {key: value.pin_memory() for key, value in features.items()}
        labels = labels.pin_memory()
    return (
        move_features_to_device(features, device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def move_tensors_to_device(tensors, device, pin_memory=False):
    non_blocking = pin_memory and device.type == "cuda"
    if pin_memory and device.type == "cuda":
        tensors = [tensor.pin_memory() for tensor in tensors]
    return [tensor.to(device, non_blocking=non_blocking) for tensor in tensors]


class TextCollator:
    def __init__(self, tokenize):
        self.tokenize = tokenize

    def __call__(self, batch):
        texts, labels = zip(*batch)
        features = self.tokenize(list(texts))
        return features, torch.stack(labels)


def embedding_collate(batch):
    embeddings, labels = zip(*batch)
    return torch.stack(embeddings), torch.stack(labels)


def make_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers, pin_memory):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(**loader_kwargs)


def copy_module_state(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_best_checkpoint(
    output_path, encoder, head, best_states, coord_mean, coord_std, args
):
    encoder_state = encoder.state_dict()
    head_state = head.state_dict()
    encoder.load_state_dict(best_states["encoder"])
    head.load_state_dict(best_states["head"])
    save_model(output_path, encoder, head, coord_mean, coord_std, args)
    encoder.load_state_dict(encoder_state)
    head.load_state_dict(head_state)


def should_save_checkpoint(epoch, total_epochs, save_every_epochs, pending_save):
    if not pending_save:
        return False
    if epoch == total_epochs:
        return True
    return epoch % save_every_epochs == 0


@torch.no_grad()
def encode_texts(encoder, texts, batch_size, device):
    encoder.eval()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        features = encoder.tokenize(batch)
        features = move_features_to_device(features, device)
        batch_embeddings = encoder(features)["sentence_embedding"]
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def train_head_epoch(head, dataloader, optimizer, loss_fn, device):
    head.train()
    total_loss = 0.0
    pin_memory = dataloader.pin_memory

    for embeddings, labels in dataloader:
        embeddings, labels = move_tensors_to_device(
            [embeddings, labels], device, pin_memory=pin_memory
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = head(embeddings)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_head(head, dataloader, loss_fn, coord_mean, coord_std, device):
    head.eval()
    total_loss = 0.0
    predictions_all = []
    labels_all = []
    pin_memory = dataloader.pin_memory

    for embeddings, labels in dataloader:
        embeddings, labels = move_tensors_to_device(
            [embeddings, labels], device, pin_memory=pin_memory
        )
        predictions = head(embeddings)
        loss = loss_fn(predictions, labels)
        total_loss += loss.item() * labels.size(0)
        predictions_all.append(predictions)
        labels_all.append(labels)

    pred_coords = torch.cat(predictions_all).float().cpu().numpy() * coord_std + coord_mean
    true_coords = torch.cat(labels_all).float().cpu().numpy() * coord_std + coord_mean
    errors_km = haversine_km(pred_coords, true_coords)

    return total_loss / len(dataloader.dataset), float(np.mean(errors_km))


def train_epoch(
    encoder,
    head,
    dataloader,
    optimizer,
    loss_fn,
    device,
    encoder_trainable,
):
    if encoder_trainable:
        encoder.train()
    else:
        encoder.eval()
    head.train()
    total_loss = 0.0
    pin_memory = dataloader.pin_memory

    for features, labels in dataloader:
        features, labels = move_batch_to_device(
            features, labels, device, pin_memory=pin_memory
        )
        optimizer.zero_grad(set_to_none=True)
        if encoder_trainable:
            embeddings = encoder(features)["sentence_embedding"]
        else:
            with torch.no_grad():
                embeddings = encoder(features)["sentence_embedding"]
        predictions = head(embeddings)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(encoder, head, dataloader, loss_fn, coord_mean, coord_std, device):
    encoder.eval()
    head.eval()
    total_loss = 0.0
    predictions_all = []
    labels_all = []
    pin_memory = dataloader.pin_memory

    for features, labels in dataloader:
        features, labels = move_batch_to_device(
            features, labels, device, pin_memory=pin_memory
        )
        embeddings = encoder(features)["sentence_embedding"]
        predictions = head(embeddings)
        loss = loss_fn(predictions, labels)
        total_loss += loss.item() * labels.size(0)
        predictions_all.append(predictions)
        labels_all.append(labels)

    pred_coords = torch.cat(predictions_all).float().cpu().numpy() * coord_std + coord_mean
    true_coords = torch.cat(labels_all).float().cpu().numpy() * coord_std + coord_mean
    errors_km = haversine_km(pred_coords, true_coords)

    return total_loss / len(dataloader.dataset), float(np.mean(errors_km))


def haversine_km(pred_coords, true_coords):
    lat1 = np.radians(pred_coords[:, 0])
    lon1 = np.radians(pred_coords[:, 1])
    lat2 = np.radians(true_coords[:, 0])
    lon2 = np.radians(true_coords[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def save_model(output_path, encoder, head, coord_mean, coord_std, args):
    os.makedirs(output_path, exist_ok=True)
    encoder.save(output_path)
    torch.save(head.state_dict(), os.path.join(output_path, "coordinate_head.pt"))
    metadata = {
        "coord_mean": coord_mean.tolist(),
        "coord_std": coord_std.tolist(),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "model_name": args.model_name,
    }
    with open(os.path.join(output_path, "coordinate_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_initial_state(output_path, encoder, head, coord_mean, coord_std, args):
    os.makedirs(output_path, exist_ok=True)
    encoder.save(os.path.join(output_path, "initial_encoder"))
    torch.save(
        head.state_dict(),
        os.path.join(output_path, "initial_coordinate_head.pt"),
    )
    metadata = {
        "coord_mean": coord_mean.tolist(),
        "coord_std": coord_std.tolist(),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "model_name": args.model_name,
    }
    with open(os.path.join(output_path, "coordinate_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def freeze_encoder_parts(encoder, args):
    if args.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        return

    transformer = encoder[0].auto_model
    if args.freeze_transformer_layers > 0:
        layers = transformer.encoder.layer[: args.freeze_transformer_layers]
        for layer in layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    if args.freeze_attention:
        for name, parameter in transformer.named_parameters():
            if ".attention." in name or name.startswith("attention."):
                parameter.requires_grad = False


def count_trainable_parameters(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def make_optimizer(encoder, head, args):
    parameter_groups = []
    encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
    if encoder_parameters:
        group = {"params": encoder_parameters, "lr": args.learning_rate}
        parameter_groups.append(group)
    parameter_groups.append(
        {"params": head.parameters(), "lr": args.head_learning_rate}
    )
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    data = pd.read_csv(args.data_file)
    data = data.dropna(subset=["text", "latitude", "longitude"])
    texts = data["text"].astype(str).tolist()
    coordinates = data[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    normalized_coordinates, coord_mean, coord_std = normalize_coordinates(coordinates)

    train_indices, val_indices = split_indices(
        data, test_size=args.test_size, seed=args.seed
    )

    train_dataset = SignCoordinateDataset(
        [texts[i] for i in train_indices], normalized_coordinates[train_indices]
    )
    val_dataset = SignCoordinateDataset(
        [texts[i] for i in val_indices], normalized_coordinates[val_indices]
    )

    encoder = SentenceTransformer(args.model_name, device=str(device))
    encoder.to(device)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    head = CoordinateRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    save_initial_state(args.output_path, encoder, head, coord_mean, coord_std, args)
    freeze_encoder_parts(encoder, args)
    encoder_trainable, encoder_total = count_trainable_parameters(encoder)
    head_trainable, head_total = count_trainable_parameters(head)
    print(
        f"Trainable encoder params: {encoder_trainable:,}/{encoder_total:,}; "
        f"head params: {head_trainable:,}/{head_total:,}"
    )

    if encoder_trainable == 0:
        print("Caching frozen encoder embeddings...")
        all_embeddings = encode_texts(encoder, texts, args.batch_size, device)
        train_dataset = EmbeddingCoordinateDataset(
            all_embeddings[train_indices], normalized_coordinates[train_indices]
        )
        val_dataset = EmbeddingCoordinateDataset(
            all_embeddings[val_indices], normalized_coordinates[val_indices]
        )
        train_loader = make_dataloader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            collate_fn=embedding_collate,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_dataloader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            collate_fn=embedding_collate,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    else:
        text_collate = TextCollator(encoder.tokenize)
        train_loader = make_dataloader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            collate_fn=text_collate,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_dataloader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            collate_fn=text_collate,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    optimizer = make_optimizer(encoder, head, args)
    loss_fn = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-7
    )
    best_val_loss = float("inf")
    best_states = None
    pending_save = False

    print(
        f"Training on {len(train_dataset):,} rows; "
        f"validating on {len(val_dataset):,} rows; "
        f"batch_size={args.batch_size}; num_workers={args.num_workers}"
    )
    for epoch in range(1, args.epochs + 1):
        if encoder_trainable == 0:
            train_loss = train_head_epoch(
                head, train_loader, optimizer, loss_fn, device
            )
            val_loss, val_error_km = evaluate_head(
                head, val_loader, loss_fn, coord_mean, coord_std, device
            )
        else:
            train_loss = train_epoch(
                encoder,
                head,
                train_loader,
                optimizer,
                loss_fn,
                device,
                encoder_trainable > 0,
            )
            val_loss, val_error_km = evaluate(
                encoder,
                head,
                val_loader,
                loss_fn,
                coord_mean,
                coord_std,
                device,
            )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]["lr"]
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_error_km={val_error_km:.3f} "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_states = {
                "encoder": copy_module_state(encoder),
                "head": copy_module_state(head),
            }
            pending_save = True

        if should_save_checkpoint(
            epoch, args.epochs, args.save_every_epochs, pending_save
        ):
            save_best_checkpoint(
                args.output_path,
                encoder,
                head,
                best_states,
                coord_mean,
                coord_std,
                args,
            )
            pending_save = False
            print(
                f"Saved best model to {args.output_path} "
                f"(val_loss={best_val_loss:.6f})"
            )


if __name__ == "__main__":
    main()
