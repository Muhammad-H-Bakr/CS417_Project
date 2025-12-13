
"""
Simplified Data Processing Script
---------------------------------
This script prepares a unified fruit freshness dataset and builds
training/validation/test generators for model training.
"""


import os
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = Path.cwd()
print(f"BASE_PATH: {BASE_PATH}")



# Each fruit maps to its rotten and fresh classes in the original dataset.
FRUIT_CATEGORIES = {
    "avocado": {
        "rotten": ["Avocado Black 2", "Avocado ripe 3"],
        "fresh": ["Avocado 1", "Avocado Black 1", "Avocado Green 1"],
    },
    "nut": {
        "rotten": ["Nut 1", "Nut 4", "Nut 5", "Nut Pecan 1"],
        "fresh": ["Nut 2", "Nut 3", "Nut Forest 1"],
    },
    "plum": {"rotten": ["Plum 3", "Plum 4"], "fresh": ["Plum 1", "Plum 2"]},
    "peach": {
        "rotten": ["Peach 3", "Peach 4", "Peach 5", "Peach 6"],
        "fresh": ["Peach 1", "Peach 2", "Peach Flat 1"],
    },
    "cactus": {
        "rotten": ["Cactus fruit green 1"],
        "fresh": ["Cactus fruit 1", "Cactus fruit red 1"],
    },
}



ORIGINAL_DATASET_PATH = os.path.join(BASE_PATH, 'fruits-360-100x100')
OUTPUT_PATH = os.path.join(BASE_PATH, 'multi_fruit_freshness_dataset')

TRAIN_DIR = os.path.join(OUTPUT_PATH, 'Training')
TEST_DIR = os.path.join(OUTPUT_PATH, 'Test')

DATA_CONFIG_FILE = 'data_config.pkl'
GENERATORS_INFO_FILE = 'generators_info.pkl'

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def ensure_dataset_exists():
    """
    Ensures the original dataset exists.
    If missing clone from git
    """
    if Path(ORIGINAL_DATASET_PATH).exists():
        print(f"Dataset found at: {ORIGINAL_DATASET_PATH}")
        return True
    else:
        print(f" Dataset not found at: {ORIGINAL_DATASET_PATH}")
        print(" Cloning fruits-360 dataset...")

        try:
            # Use subprocess to clone the repository
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/fruits-360/fruits-360-100x100.git",
                ],
                cwd=BASE_PATH,  # Set working directory
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"Dataset cloned successfully to: {ORIGINAL_DATASET_PATH}")
                return True
            else:
                print(f"Failed to clone dataset. Error: {result.stderr}")
                return False

        except Exception as e:
            print(f" Error cloning dataset: {e}")
            return False
