import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import pandas as pd
import albumentations as A
import numpy as np
import ast

# Import our data pipeline functions
from data_pipeline import get_tf_dataset

IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    # 1. Load the manifest
    print("[OPERATION] Loading manifest...")
    df = pd.read_csv("manifest.csv")
    df["concept_vec"] = df["concept_vec"].apply(ast.literal_eval)
    
    # 2. Define our aggressive augmentations via albumentations
    # Sabotage the training data so it cannot be memorized
    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(10, 30), hole_width_range=(10, 30), p=0.5)
    ])
    
    def train_aug_fn(img_tensor):
        # convert tensor to numpy, apply A, convert back
        def _aug(img):
            return transform_train(image=img)['image'].astype(np.float32)
        # tf.numpy_function needs to know output types
        out = tf.numpy_function(_aug, [img_tensor], tf.float32)
        out.set_shape(img_tensor.shape)
        return out
        
    print("[OPERATION] Compiling tf.data pipelines...")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    train_ds = get_tf_dataset(train_df, "train", batch_size=BATCH_SIZE, shuffle=True, augment_fn=train_aug_fn)
    val_ds = get_tf_dataset(val_df, "val", batch_size=BATCH_SIZE, shuffle=False, augment_fn=None)
    
    # 3. Build Architecture
    print("[OPERATION] Building Architecture...")
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.75
    )
    base_model.trainable = True
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.40)(x)
    
    c_out = Dense(4, activation='sigmoid', name='C', kernel_regularizer=l2(0.01))(x)
    combined_features = Concatenate()([x, c_out])
    d_out = Dense(15, activation='softmax', name='D', kernel_regularizer=l2(0.01))(combined_features)
    
    model = Model(inputs=base_model.input, outputs=[c_out, d_out])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
        loss_weights=[0.4, 1.0],
        metrics=[["accuracy"], ["accuracy"]]
    )
    
    # 4. Train with EarlyStopping and ModelCheckpoint
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_D_accuracy', mode='max', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_D_accuracy', mode='max', save_best_only=True)
    ]
    
    print("[OPERATION] Training MVP Model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30, # Increased epochs since we have EarlyStopping
        callbacks=callbacks
    )
    
    print("\n[SUCCESS] Master model trained and saved as best_model.keras")

if __name__ == "__main__":
    main()
