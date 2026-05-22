"""
Module 5: Video-Level Prediction (CORRECTED FOR XCEPTION)
Purpose: Predict Real/Fake for WHOLE VIDEOS using the Xception model
Input : detected_faces/real/<video_folder> , detected_faces/fake/<video_folder>
Output: Final Video Classification Report
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from tensorflow.keras.applications.xception import preprocess_input # Mukkiyamaana Import

# ==========================================================
# VIDEO PREDICTOR CLASS
# ==========================================================
class VideoPredictor:
    def __init__(self, model_path, img_size=(224, 224), confidence_threshold=0.6):
        self.model_path = model_path
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # NOTE: Check your Module 4 classes. Usually ['FAKE', 'REAL'] based on alphabet.
        # If accuracy is 0%, swap this to ["REAL", "FAKE"]
        self.class_names = ["FAKE", "REAL"]

        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"🔧 Loading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("❌ Trained model not found. Run Module 4 first.")

        self.model = keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully")

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 1. Resize
        img = cv2.resize(img, self.img_size)

        # 2. BGR to RGB (for Xception)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Xception Preprocessing (Don't use / 255.0)
        img = img.astype("float32")
        img = preprocess_input(img)

        # 4. Add Batch Dimension
        img = np.expand_dims(img, axis=0)
        return img

    def predict_frames(self, video_folder):
        face_files = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not face_files:
            return []

        predictions = []

        for file in face_files:
            path = os.path.join(video_folder, file)
            img = self.preprocess_image(path)

            if img is not None:
                preds = self.model.predict(img, verbose=0)[0]

                # Argmax eduthu Class kandupidippom
                pred_class = np.argmax(preds)
                confidence = np.max(preds)
                label = self.class_names[pred_class]

                # Threshold Check
                if confidence < self.confidence_threshold:
                    label = "UNCERTAIN"

                predictions.append({
                    "filename": file,
                    "label": label,
                    "confidence": confidence
                })

        return predictions

    def aggregate_video_prediction(self, frame_predictions):
        valid_preds = [p for p in frame_predictions if p["label"] != "UNCERTAIN"]

        if not valid_preds:
            return "UNCERTAIN", 0.0

        labels = [p["label"] for p in valid_preds]
        counts = Counter(labels)

        fake_count = counts.get("FAKE", 0)
        real_count = counts.get("REAL", 0)
        total = len(valid_preds)

        # Majority Vote
        if fake_count > real_count:
            final_label = "FAKE"
            final_conf = fake_count / total
        else:
            final_label = "REAL"
            final_conf = real_count / total

        return final_label, final_conf


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION - MODULE 5 (XCEPTION UPDATE)")
    print("=" * 60)

    MODEL_PATH = "models/deepfake_model.keras"
    DATASET_ROOT = "detected_faces"

    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Dataset not found: {DATASET_ROOT}")
        return

    try:
        predictor = VideoPredictor(model_path=MODEL_PATH)
    except Exception as e:
        print(e)
        return

    total_videos = 0
    correct_predictions = 0

    # Folders Check
    for true_label in ["real", "fake"]:
        input_dir = os.path.join(DATASET_ROOT, true_label)
        if not os.path.exists(input_dir):
            continue

        video_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        print(f"\n🎥 Evaluating {true_label.upper()} videos ({len(video_folders)})...")

        for video_name in video_folders:
            video_path = os.path.join(input_dir, video_name)

            # Predict
            frame_preds = predictor.predict_frames(video_path)

            if not frame_preds:
                continue

            pred_label, conf = predictor.aggregate_video_prediction(frame_preds)

            # Check correctness
            is_correct = (pred_label.lower() == true_label)
            if is_correct:
                correct_predictions += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f"  {status} | {video_name:<20} -> Predicted: {pred_label} ({conf:.1%})")
            total_videos += 1

    print("\n" + "=" * 60)
    if total_videos > 0:
        accuracy = (correct_predictions / total_videos) * 100
        print(f"TOTAL VIDEOS      : {total_videos}")
        print(f"CORRECT PREDICTED : {correct_predictions}")
        print(f"VIDEO ACCURACY    : {accuracy:.2f}%")
    else:
        print("No videos found.")
    print("=" * 60)


if __name__ == "__main__":
    main()