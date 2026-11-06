import tensorflow as tf
import pandas as pd
from data_pipeline import get_tf_dataset
import ast

def main():
    print("[OPERATION] Loading model...")
    model = tf.keras.models.load_model("best_model.keras")

    print("[OPERATION] Loading manifest for clean calibration data...")
    df = pd.read_csv("manifest.csv")
    df["concept_vec"] = df["concept_vec"].apply(ast.literal_eval)
    train_df = df[df["split"] == "train"]

    # Use clean images (augment_fn=None) for calibration
    train_ds = get_tf_dataset(train_df, "train", batch_size=32, shuffle=True, augment_fn=None)

    def representative_data_gen():
        # Take 100 clean images
        for input_value, _ in train_ds.unbatch().take(100):
            yield [tf.expand_dims(input_value, 0)]

    print("[OPERATION] Initiating INT8 Post-Training Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open('edge_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("[SUCCESS] Quantized model saved to edge_model.tflite")

if __name__ == "__main__":
    main()
