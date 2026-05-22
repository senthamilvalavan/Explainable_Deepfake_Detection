"""
Module 1: Video to Frame Extraction (TRAINING VERSION)
Purpose: Extract frames from dataset videos at 1 FPS
Input : Dataset/real , Dataset/fake (videos)
Output: extracted_frames/real/<video_name>/, extracted_frames/fake/<video_name>/
"""

import cv2
import os
import sys
import traceback


class VideoFrameExtractor:
    def __init__(self, video_path, output_folder, fps=1):
        self.video_path = video_path
        self.output_folder = output_folder
        self.target_fps = fps

    def extract_frames(self):
        # Video file irukannu check panrom
        if not os.path.exists(self.video_path):
            print(f"✗ Video not found: {self.video_path}")
            return 0

        # Output folder illana create panrom
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"✗ Cannot open video: {self.video_path}")
            return 0

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # 0 division error varama iruka check
        if video_fps <= 0:
            frame_interval = 1
        else:
            frame_interval = int(video_fps / self.target_fps)

        frame_interval = max(1, frame_interval)

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame interval calculation
            if frame_count % frame_interval == 0:
                filename = f"frame_{saved_count + 1:04d}.jpg"
                out_path = os.path.join(self.output_folder, filename)
                cv2.imwrite(out_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        return saved_count


def main():
    print("\n" + "=" * 60)
    print("DEEPFAKE DETECTION - MODULE 1 (TRAINING)")
    print("=" * 60)

    # Folder Settings
    DATASET_ROOT = "Dataset"
    OUTPUT_ROOT = "extracted_frames"
    FPS = 1  # 1 frame per second

    # Dataset folder iruka nu check panrom
    if not os.path.exists(DATASET_ROOT):
        print("✗ Dataset folder not found!")
        print("Expected structure: Dataset/real and Dataset/fake")
        # Folder illana create panna solli warning
        try:
            os.makedirs(os.path.join(DATASET_ROOT, "real"))
            os.makedirs(os.path.join(DATASET_ROOT, "fake"))
            print("✔ Created empty Dataset folders. Please add videos there and run again.")
        except:
            pass
        return

    total_frames = 0

    # Real mattrum Fake folder-a process panrom
    for label in ["real", "fake"]:
        input_dir = os.path.join(DATASET_ROOT, label)
        base_output_dir = os.path.join(OUTPUT_ROOT, label)

        if not os.path.exists(input_dir):
            print(f"⚠ Skipping missing folder: {input_dir}")
            continue

        # Supported video formats
        video_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        ]

        if len(video_files) == 0:
            print(f"⚠ No videos found in {input_dir}")
            continue

        print(f"\n▶ Processing {label.upper()} videos ({len(video_files)})")

        for video in video_files:
            video_path = os.path.join(input_dir, video)

            # --- CHANGE IS HERE ---
            # Video perai vachu thaniya folder create panrom
            video_name = os.path.splitext(video)[0]
            video_specific_output_dir = os.path.join(base_output_dir, video_name)

            print(f"  🎬 {video} -> {video_specific_output_dir}")

            extractor = VideoFrameExtractor(
                video_path=video_path,
                output_folder=video_specific_output_dir,
                fps=FPS
            )

            frames = extractor.extract_frames()
            total_frames += frames
            print(f"     ✓ Extracted {frames} frames")

    print("\n" + "=" * 60)
    print("MODULE 1 COMPLETED")
    print(f"Total frames extracted: {total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\n✗ Unexpected error:", e)
        traceback.print_exc()
        sys.exit(1)