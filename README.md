# Sign Coordinate Regressor

This project fine-tunes a sentence-transformer model to predict a latitude and
longitude pair from text observed on signs at an intersection.

The raw dataset is expected at `training_data_raw.csv`. It may include the
pandas-exported index column; the preprocessing script only uses:

- `intersection`
- `text_on_sign_exact`
- `latitude`
- `longitude`

## Workflow

Prepare bootstrapped training rows:

```bash
source .venv/bin/activate
python prepare_training_data.py --seed 1992 --bag-size 5 --samples-per-intersection 50
```

This writes `training.csv` with rows shaped like:

```text
intersection,sample_id,text,latitude,longitude,unique_sign_count,raw_sign_count
```

Each row is a deterministic bootstrap sample of sign texts from one
intersection, joined into a single text field. This trains on "some signs seen
at this coordinate" instead of one sign or every sign at that coordinate.

Train the model:

```bash
python train.py
```

Evaluate and write predictions:

```bash
python eval.py
```

This writes `predictions.csv` and a map-style diagnostic plot:

![Prediction map](./plots/prediction_map.png)

It also writes a coordinate calibration plot:

![Predicted vs actual coordinates](./plots/predicted_vs_actual.png)

Or run the full pipeline:

```bash
make
```

## Useful Options

`prepare_training_data.py`:

- `--seed`: deterministic bootstrap seed.
- `--bag-size`: number of sign texts per sampled bag.
- `--samples-per-intersection`: number of bags generated per intersection.

`train.py`:

- `--model-name`: sentence-transformers base model.
- `--epochs`: training epochs.
- `--batch-size`: batch size.
- `--device`: device override such as `cpu`, `cuda`, or `mps`. Defaults to `cuda`.

## Outputs

- `training.csv`: prepared bootstrapped dataset.
- `output/`: saved sentence-transformer encoder, coordinate head, and coordinate
  normalization metadata.
- `predictions.csv`: evaluation rows with predicted coordinates and `error_km`.
- `plots/prediction_map.png`: actual vs predicted coordinates with line segments
  showing the prediction error.
- `plots/predicted_vs_actual.png`: predicted vs actual latitude and longitude
  scatter plots.
