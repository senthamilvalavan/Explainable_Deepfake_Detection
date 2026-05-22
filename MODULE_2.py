"""
Module 2: Face Detection and Cropping (TRAINING VERSION)
Purpose: Detect and crop faces from extracted frames
Input : extracted_frames/real/<video_folder> , extracted_frames/fake/<video_folder>
Output: detected_faces/real/<video_folder> , detected_faces/fake/<video_folder>
"""

import cv2
import os
import sys
import traceback

# Try importing MTCNN
try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("⚠ Warning: MTCNN not available. Falling back to Haar Cascade")


class FaceDetector:
    def __init__(self, method="mtcnn", min_face_size=(80, 80), padding=20):
        self.method = method.lower()
        self.min_face_size = min_face_size
        self.padding = padding
        self.faces_detected = 0
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if self.method == "mtcnn" and MTCNN_AVAILABLE:
            print("✓ Using MTCNN face detector")
            self.detector = MTCNN(min_face_size=min(self.min_face_size))
        else:
            print("✓ Using Haar Cascade face detector")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        if self.method == "mtcnn" and MTCNN_AVAILABLE:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb)
            faces = []
            for d in detections:
                x, y, w, h = d["box"]
                conf = d["confidence"]
                if conf > 0.9 and w >= self.min_face_size[0] and h >= self.min_face_size[1]:
                    faces.append((max(0, x), max(0, y), w, h))
            return faces
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size
            )
            return faces

    def crop_face(self, image, box):
        x, y, w, h = box
        h_img, w_img = image.shape[:2]

        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(w_img, x + w + self.padding)
        y2 = min(h_img, y + h + self.padding)

        return image[y1:y2, x1:x2]

    def process_folder(self, frames_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        frame_files = sorted(
            f for f in os.listdir(frames_folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )

        if len(frame_files) == 0:
            # print(f"⚠ No frames in {frames_folder}") # Too much noise if empty
            return

        print(f"  📂 Processing: {os.path.basename(frames_folder)}")

        for idx, frame_file in enumerate(frame_files, 1):
            frame_path = os.path.join(frames_folder, frame_file)
            image = cv2.imread(frame_path)

            if image is None:
                continue

            faces = self.detect_faces(image)

            for i, box in enumerate(faces, 1):
                cropped = self.crop_face(image, box)
                # Video peyaraiyum file name laye vachu save panrom
                out_name = f"{os.path.splitext(frame_file)[0]}_face_{i}.jpg"
                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, cropped)
                self.faces_detected += 1

            # Oru line la status theriyum (Optional cleanup)
            # print(f"[{idx}/{len(frame_files)}] {frame_file} → {len(faces)} face(s)")


def main():
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION - MODULE 2 (TRAINING)")
    print("=" * 60)

    FRAME_ROOT = "extracted_frames"
    OUTPUT_ROOT = "detected_faces"

    if not os.path.exists(FRAME_ROOT):
        print("✗ extracted_frames folder not found. Run Module-1 first.")
        return

    detector = FaceDetector(
        method="mtcnn",
        min_face_size=(80, 80),
        padding=20
    )

    for label in ["real", "fake"]:
        base_input_path = os.path.join(FRAME_ROOT, label)
        base_output_path = os.path.join(OUTPUT_ROOT, label)

        if not os.path.exists(base_input_path):
            print(f"⚠ Skipping missing folder: {base_input_path}")
            continue

        print(f"\n▶ Processing {label.upper()} Data...")

        # --- IMPORTANT CHANGE IS HERE ---
        # Iterate through VIDEO FOLDERS inside extracted_frames/real/

        video_folders = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]

        if not video_folders:
            print(f"⚠ No video folders found in {base_input_path}")
            continue

        for video_folder in video_folders:
            # Full path to the video's frame folder
            current_frame_dir = os.path.join(base_input_path, video_folder)

            # Create a matching output folder for faces
            current_output_dir = os.path.join(base_output_path, video_folder)

            detector.process_folder(current_frame_dir, current_output_dir)

    print("\n" + "=" * 60)
    print("MODULE 2 COMPLETED")
    print(f"Total faces detected: {detector.faces_detected}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\nUnexpected error:", e)
        traceback.print_exc()
        sys.exit(1)