"""
Module 6: Visual Evidence Generator (FINAL VERDICT MODE)
Purpose:
  1. Side-by-Side Evidence (Multi-color for Fake, Blue for Real)
  2. Final Video Summary (Percentage of Real vs Fake)

"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

# ================= CONFIGURATION =================
INPUT_FOLDER = "Test_Videos"
OUTPUT_FOLDER = "evidence_output_final"
MODEL_PATH = "models/deepfake_model.keras"
MAX_HEATMAPS_PER_VIDEO = 20
# =================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 1. Heatmap Generator (Gradient Calculation) ---
def generate_heatmap_split_method(img_array, model):
    try:
        base_model = model.layers[0]
        head_layers = model.layers[1:]

        with tf.GradientTape() as tape:
            conv_outputs = base_model(img_array, training=False)
            tape.watch(conv_outputs)
            preds = conv_outputs
            for layer in head_layers:
                preds = layer(preds, training=False)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap
    except Exception as e:
        print(f"      ⚠️ Heatmap Error: {e}")
        return None

# --- 2. Overlay Functions ---
def overlay_heatmap_jet(img, heatmap, alpha=0.5): # For FAKE
    if heatmap is None: return img
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

def overlay_heatmap_blue(img, heatmap, alpha=0.5): # For REAL
    if heatmap is None: return img
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_WINTER)
    return cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

# --- 3. Main Processing Logic ---
def process_video_with_heatmap(video_path, model):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(OUTPUT_FOLDER, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    print(f"\n🎥 Processing: {video_name}...")

    frame_count = 0
    saved_count = 0

    # VOTING COUNTERS
    real_votes = 0
    fake_votes = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Check every 15th frame
        if frame_count % 15 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            for i, (x, y, w, h) in enumerate(faces):
                # Count votes even if we stop saving images
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_float = face_rgb.astype("float32")
                face_batch = np.expand_dims(preprocess_input(face_float), axis=0)

                # Predict
                preds = model.predict(face_batch, verbose=0)
                pred_idx = np.argmax(preds[0])
                conf = np.max(preds[0])

                label = "FAKE" if pred_idx == 0 else "REAL"

                # --- VOTE ---
                if label == "FAKE":
                    fake_votes += 1
                else:
                    real_votes += 1

                # Save Evidence (Limit to MAX)
                if saved_count < MAX_HEATMAPS_PER_VIDEO:
                    heatmap = generate_heatmap_split_method(face_batch, model)

                    if heatmap is not None:
                        if label == "FAKE":
                            overlay_img = overlay_heatmap_jet(face_resized, heatmap)
                            status_msg = "🔴 FAKE Evidence Saved"
                        else:
                            overlay_img = overlay_heatmap_blue(face_resized, heatmap)
                            status_msg = "🔵 REAL Evidence Saved"

                        final_img = np.hstack([face_resized, overlay_img])
                    else:
                        final_img = np.hstack([face_resized, face_resized])
                        status_msg = "⚠️ No Heatmap"

                    filename = f"frame_{frame_count}_{label}_{conf:.2f}.jpg"
                    save_path = os.path.join(save_dir, filename)

                    if cv2.imwrite(save_path, final_img):
                        print(f"   {status_msg}: {filename}")
                        saved_count += 1

        frame_count += 1
    cap.release()

    # --- 4. FINAL VERDICT CALCULATION ---
    total_votes = real_votes + fake_votes

    print("-" * 40)
    if total_votes == 0:
        print(f"⚠️ No Faces Found in {video_name}")
    else:
        fake_percent = (fake_votes / total_votes) * 100
        real_percent = (real_votes / total_votes) * 100

        # Majority Rule
        if fake_percent > 50: # More than 50% Fake frames
            final_verdict = "FAKE"
            emoji = "🚨"
        else:
            final_verdict = "REAL"
            emoji = "✅"

        print(f"SUMMARY FOR: {video_name}")
        print(f"   Total Faces Scanned : {total_votes}")
        print(f"   Fake Confidence     : {fake_percent:.1f}%")
        print(f"   Real Confidence     : {real_percent:.1f}%")
        print(f"   ---------------------------")
        print(f"   🏁 FINAL RESULT     : {emoji} {final_verdict}")
    print("-" * 40 + "\n")

# ================= MAIN =================
if __name__ == "__main__":
    print("="*60)
    print("MODULE 8: FINAL VERDICT SYSTEM")
    print("="*60)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model missing: {MODEL_PATH}")
    else:
        model = keras.models.load_model(MODEL_PATH)
        if not os.path.exists(INPUT_FOLDER):
            os.makedirs(INPUT_FOLDER)
            print(f"📁 Created '{INPUT_FOLDER}'. Add videos and run again.")
        else:
            videos = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov"))]
            for video in videos:
                process_video_with_heatmap(os.path.join(INPUT_FOLDER, video), model)
    print(f"✅ All Done.")