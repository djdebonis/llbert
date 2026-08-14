all: data train eval

training.csv: prepare_training_data.py training_data_raw.csv
	@echo "Preparing bootstrapped sign text bags..."
	@bash -c 'source .venv/bin/activate && python prepare_training_data.py'

data: training.csv

train: train.py training.csv
	@echo "Training coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python train.py --epochs 50'

eval: eval.py training.csv
	@echo "Evaluating coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python eval.py'

train_frozen_encoder: train.py training.csv
	@echo "Training coordinate head with frozen encoder..."
	@bash -c 'source .venv/bin/activate && python train.py --output-path output_frozen_encoder --freeze-encoder --epochs 10'

eval_frozen_encoder: eval.py training.csv
	@echo "Evaluating frozen-encoder coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python eval.py --model-path output_frozen_encoder --output-file predictions_frozen_encoder.csv --plot-file plots/prediction_map_frozen_encoder.png --scatter-plot-file plots/predicted_vs_actual_frozen_encoder.png'

train_frozen_layers: train.py training.csv
	@echo "Training coordinate regressor with first transformer layers frozen..."
	@bash -c 'source .venv/bin/activate && python train.py --output-path output_frozen_layers --freeze-transformer-layers 4 --epochs 50 --num-workers 0'

eval_frozen_layers: eval.py training.csv
	@echo "Evaluating frozen-layer coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python eval.py --model-path output_frozen_layers --output-file predictions_frozen_layers.csv --plot-file plots/prediction_map_frozen_layers.png --scatter-plot-file plots/predicted_vs_actual_frozen_layers.png'

train_mpnet: train.py training.csv
	@echo "Training with all-mpnet-base-v2 (encoder fully frozen, head only)..."
	@bash -c 'source .venv/bin/activate && python train.py \
		--model-name sentence-transformers/all-mpnet-base-v2 \
		--output-path output_mpnet \
		--freeze-encoder \
		--hidden-dim 512 \
		--head-learning-rate 1e-2 \
		--epochs 50'

train_mpnet_finetune: train.py training.csv
	@echo "Training with all-mpnet-base-v2 (frozen first 10 of 12 layers)..."
	@bash -c 'source .venv/bin/activate && python train.py \
		--model-name sentence-transformers/all-mpnet-base-v2 \
		--output-path output_mpnet_ft \
		--freeze-transformer-layers 10 \
		--hidden-dim 512 \
		--head-learning-rate 1e-2 \
		--batch-size 32 \
		--epochs 50'

eval_mpnet_finetune: eval.py training.csv
	@echo "Evaluating all-mpnet-base-v2 fine-tuned coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python eval.py \
		--model-path output_mpnet_ft \
		--output-file predictions_mpnet_ft.csv \
		--plot-file plots/prediction_map_mpnet_ft.png \
		--scatter-plot-file plots/predicted_vs_actual_mpnet_ft.png'

eval_mpnet: eval.py training.csv
	@echo "Evaluating all-mpnet-base-v2 coordinate regressor..."
	@bash -c 'source .venv/bin/activate && python eval.py \
		--model-path output_mpnet \
		--output-file predictions_mpnet.csv \
		--plot-file plots/prediction_map_mpnet.png \
		--scatter-plot-file plots/predicted_vs_actual_mpnet.png'

lint:
	@echo "Auto-linting files and performing final style checks..."
	@bash -c 'source .venv/bin/activate && isort --profile=black *.py'
	@bash -c 'source .venv/bin/activate && black *.py'
	@bash -c 'source .venv/bin/activate && flake8 --max-line-length=88 --ignore E203 *.py'

clean:
	@echo "Removing generated outputs"
	@rm -rf output/
	@rm -f training.csv predictions.csv

.PHONY: data train eval train_frozen_encoder eval_frozen_encoder train_frozen_layers eval_frozen_layers train_mpnet eval_mpnet train_mpnet_finetune eval_mpnet_finetune lint clean all
