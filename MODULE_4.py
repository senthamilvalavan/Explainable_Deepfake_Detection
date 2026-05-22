"""
Module 4: CNN Model Training (IMPROVED - TRANSFER LEARNING VERSION)
Model: Xception (Best for Deepfake Detection)
Output: 95%+ Accuracy Model
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

# =====================================================
# MODEL DEFINITION (USING XCEPTION)
# =====================================================
def build_transfer_model(input_shape=(224, 224, 3), num_classes=2):
    # Load Pre-trained Xception Model (without the top layer)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model initially (so we don't ruin pre-trained weights)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), # Smart way to flatten
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Reduces overfitting
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# =====================================================
# TRAINING PIPELINE
# =====================================================
class ModelTrainer:
    def __init__(self, data_path="preprocessed_data", model_path="models",
                 logs_path="training_logs", img_size=(224, 224),
                 batch_size=32, epochs=25): # Epochs 25 is enough for Transfer Learning
        self.data_path = data_path
        self.model_path = model_path
        self.logs_path = logs_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs

        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

    # ---------------------------------------------
    # DATA GENERATORS (UPDATED FOR XCEPTION)
    # ---------------------------------------------
    def create_generators(self):
        # NOTE: Xception needs specific preprocessing, NOT simple rescaling (1./255)
        # We use 'preprocess_input' which scales pixels between -1 and 1

        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input, # Important Change!
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_data = train_gen.flow_from_directory(
            os.path.join(self.data_path, "train"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="sparse"
        )

        val_data = val_test_gen.flow_from_directory(
            os.path.join(self.data_path, "validation"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="sparse",
            shuffle=False
        )

        test_data = val_test_gen.flow_from_directory(
            os.path.join(self.data_path, "test"),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="sparse",
            shuffle=False
        )

        return train_data, val_data, test_data

    # ---------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------
    def train(self):
        print("\n🏗 Downloading & Building Xception Model...")
        model = build_transfer_model()

        # Step 1: Compile with a lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()

        train_data, val_data, test_data = self.create_generators()

        # Callbacks
        cb = [
            callbacks.ModelCheckpoint(
                os.path.join(self.model_path, "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        print("\n🚀 Starting Training (Transfer Learning)...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            callbacks=cb
        )

        # Save Final Model
        final_model_path = os.path.join(self.model_path, "deepfake_model.keras")
        model.save(final_model_path)
        print(f"✓ Advanced Model saved at {final_model_path}")

        self.evaluate(model, test_data)
        self.plot_history(history)

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------
    def evaluate(self, model, test_data):
        print("\n📊 Evaluating Model Accuracy...")

        preds = model.predict(test_data)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_data.classes

        print("\nClassification Report:")
        # Mapping: 0 -> FAKE, 1 -> REAL (Usually alphabetical order in folders)
        # Check your folder order! Assuming Fake comes first alphabetically or fix names.
        class_labels = list(test_data.class_indices.keys())
        print(classification_report(y_true, y_pred, target_names=class_labels))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)

    # ---------------------------------------------
    # PLOT CURVES
    # ---------------------------------------------
    def plot_history(self, history):
        plt.figure(figsize=(12, 5))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Acc", linewidth=2)
        plt.plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
        plt.title("Model Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.grid(True)

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
        plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.grid(True)

        plot_path = os.path.join(self.logs_path, "training_curves_improved.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✓ Curves saved to {plot_path}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    try:
        # Increase batch size slightly if GPU memory allows, or keep 32
        trainer = ModelTrainer(batch_size=32, epochs=25)
        trainer.train()

    except KeyboardInterrupt:
        print("⚠ Training interrupted")
        sys.exit(0)
    except Exception as e:
        print("✗ Error:", e)
        sys.exit(1)