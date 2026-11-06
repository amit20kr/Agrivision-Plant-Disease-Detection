python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test > eval_clean.txt
python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --corrupt > eval_corrupt.txt
python scripts/quantize.py
python scripts/evaluate.py --model best_model.keras --manifest manifest.csv --split test --tflite edge_model.tflite > eval_tflite.txt
python scripts/gradcam.py --inspect best_model.keras > gradcam_inspect.txt
