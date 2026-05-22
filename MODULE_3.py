"""
Module 3: Dataset Preprocessing (TRAINING VERSION)
Purpose: Load, resize, split, and prepare FACE IMAGES for CNN training
Input : detected_faces/real/<video_folders> , detected_faces/fake/<video_folders>
Output: preprocessed_data/train|validation|test/real|fake
"""

import cv2
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class DatasetPreprocessor:
    def __init__(
            self,
            dataset_path="detected_faces",
            output_path="preprocessed_data",
            metadata_path="metadata",
            img_size=(224, 224),
            test_split=0.15,
            val_split=0.15,
            random_state=42
    ):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.metadata_path = metadata_path
        self.img_size = img_size
        self.test_split = test_split
        self.val_split = val_split
        self.random_state = random_state

        # Label mapping
        self.label_mapping = {"real": 0, "fake": 1}

        # Stats
        self.stats = {
            "total_images": 0,
            "real_images": 0,
            "fake_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "corrupted_images": 0
        }

    # --------------------------------------------------
    # Helper to count images recursively
    # --------------------------------------------------
    def _count_images(self, folder):
        count = 0
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    count += 1
        return count

    # --------------------------------------------------
    # Folder creation
    # --------------------------------------------------
    def create_folders(self):
        folders = [
            self.output_path,
            self.metadata_path,
            os.path.join(self.output_path, "train", "real"),
            os.path.join(self.output_path, "train", "fake"),
            os.path.join(self.output_path, "validation", "real"),
            os.path.join(self.output_path, "validation", "fake"),
            os.path.join(self.output_path, "test", "real"),
            os.path.join(self.output_path, "test", "fake"),
        ]
        for f in folders:
            os.makedirs(f, exist_ok=True)

    # --------------------------------------------------
    # Dataset validation
    # --------------------------------------------------
    def validate_dataset(self):
        print("\n🔍 Validating detected_faces dataset...")

        real_dir = os.path.join(self.dataset_path, "real")
        fake_dir = os.path.join(self.dataset_path, "fake")

        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print("✗ detected_faces/real or detected_faces/fake not found")
            print("  Run Module-2 (training version) first.")
            return False

        # --- CHANGE: Recursive counting ---
        self.stats["real_images"] = self._count_images(real_dir)
        self.stats["fake_images"] = self._count_images(fake_dir)
        self.stats["total_images"] = self.stats["real_images"] + self.stats["fake_images"]

        if self.stats["total_images"] == 0:
            print("✗ No images found in detected_faces folders")
            return False

        print(f"✓ Real faces : {self.stats['real_images']}")
        print(f"✓ Fake faces : {self.stats['fake_images']}")
        print(f"✓ Total      : {self.stats['total_images']}")

        return True

    # --------------------------------------------------
    # Load dataset info
    # --------------------------------------------------
    def collect_data(self):
        print("\n📂 Collecting image paths...")
        data = []

        for label_name, label_value in self.label_mapping.items():
            base_folder = os.path.join(self.dataset_path, label_name)

            # --- CHANGE: os.walk used instead of os.listdir ---
            # Ithu sub-folders kulla irukura images-ayum edukkum
            for root, dirs, files in os.walk(base_folder):
                for img in files:
                    if img.lower().endswith((".jpg", ".png", ".jpeg")):
                        data.append({
                            "image_path": os.path.join(root, img),
                            "label": label_value,
                            "label_name": label_name
                        })

        df = pd.DataFrame(data)

        # Shuffle ensures mix of videos
        df = shuffle(df, random_state=self.random_state).reset_index(drop=True)
        print(f"✓ Collected {len(df)} images for processing")
        return df

    # --------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------
    def preprocess_image(self, path):
        img = cv2.imread(path)
        if img is None:
            self.stats["corrupted_images"] += 1
            return None

        # Resize
        img = cv2.resize(img, self.img_size)

        # Optional: Normalize pixel values (0-1) - uncomment if saving as numpy array
        # img = img / 255.0

        return img

    # --------------------------------------------------
    # Split dataset
    # --------------------------------------------------
    def split_dataset(self, df):
        # First split: Train vs (Val + Test)
        train_val, test = train_test_split(
            df,
            test_size=self.test_split,
            stratify=df["label"],
            random_state=self.random_state
        )

        # Second split: Val vs Test
        # Adjust split ratio relative to remaining data
        val_ratio = self.val_split / (1 - self.test_split)

        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=train_val["label"],
            random_state=self.random_state
        )

        self.stats["train_images"] = len(train)
        self.stats["val_images"] = len(val)
        self.stats["test_images"] = len(test)

        return train, val, test

    # --------------------------------------------------
    # Save processed images
    # --------------------------------------------------
    def save_split(self, df, split_name):
        print(f"\n🔄 Processing {split_name} images...")
        count = 0

        # Create progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split_name}"):
            img = self.preprocess_image(row["image_path"])
            if img is None:
                continue

            # Save in flat structure: preprocessed_data/train/real/img_001.jpg
            out_dir = os.path.join(self.output_path, split_name, row["label_name"])
            filename = f"{split_name}_{idx:06d}.jpg"
            out_path = os.path.join(out_dir, filename)

            cv2.imwrite(out_path, img)
            count += 1

        print(f"✓ Saved {count} images to {split_name}")

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------
    def save_metadata(self):
        config = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": self.img_size,
            "test_split": self.test_split,
            "val_split": self.val_split,
            "statistics": self.stats
        }

        with open(os.path.join(self.metadata_path, "preprocessing_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print(f"\n📝 Metadata saved to {self.metadata_path}")

    # --------------------------------------------------
    # Run pipeline
    # --------------------------------------------------
    def run(self):
        print("\n" + "=" * 60)
        print("DEEPFAKE DETECTION - MODULE 3 (TRAINING)")
        print("=" * 60)

        if not self.validate_dataset():
            return

        self.create_folders()

        df = self.collect_data()

        if len(df) == 0:
            print("✗ No images collected. Check your paths.")
            return

        train_df, val_df, test_df = self.split_dataset(df)

        self.save_split(train_df, "train")
        self.save_split(val_df, "validation")
        self.save_split(test_df, "test")

        self.save_metadata()

        print("\n" + "=" * 60)
        print("MODULE 3 COMPLETED")
        print("Stats Summary:")
        print(json.dumps(self.stats, indent=2))
        print("=" * 60)


if __name__ == "__main__":
    try:
        processor = DatasetPreprocessor()
        processor.run()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\n✗ Unexpected error:", e)
        # traceback.print_exc() # Uncomment for debugging
        sys.exit(1)