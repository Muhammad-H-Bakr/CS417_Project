"""
evaluation_testing.py
---------------------
Comprehensive evaluation and testing of the trained fruit freshness model.
Inputs: Trained model and data configuration
Outputs: Metrics, confusion matrix, classification report, and visualizations
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


print("FRUIT FRESHNESS CLASSIFICATION - EVALUATION & TESTING")


# CONFIGURATION

BASE_PATH = Path.cwd()
DATA_CONFIG_FILE = "data_config.pkl"
MODEL_PATH = "best_fruit_freshness_model.keras"
HISTORY_PATH = "training_history.pkl"

print(f" Working directory: {BASE_PATH}")

# LOAD CONFIGURATION & MODEL


def load_data_config():
    """Load data configuration."""
    if not os.path.exists(DATA_CONFIG_FILE):
        raise FileNotFoundError(
            f" {DATA_CONFIG_FILE} not found! Please run 'data_processing.py' first."
        )

    with open(DATA_CONFIG_FILE, "rb") as f:
        config = pickle.load(f)

    print("\n Data configuration loaded")
    return config


def load_trained_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f" {MODEL_PATH} not found! Please run 'model_training.py' first."
        )

    model = keras.models.load_model(MODEL_PATH)
    print(f"\n Model loaded from: {MODEL_PATH}")
    return model


def load_training_history():
    """Load training history if available."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        print(f" Training history loaded from: {HISTORY_PATH}")
        return history
    else:
        print("  Training history not found")
        return None


# RECREATE TEST GENERATOR


def recreate_test_generator(config):
    """Recreate test data generator."""
    print("\n Recreating test generator...")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        config["test_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )

    print(f" Test generator created: {test_generator.samples} samples")
    return test_generator


# MODEL EVALUATION
def evaluate_model(model, test_generator):
    """Comprehensive evaluation of model on test set."""
    print("\n EVALUATING MODEL ON TEST SET")

    test_generator.reset()
    results = model.evaluate(test_generator, verbose=1)


    loss = results[0]
    accuracy = results[1]

    # Handle cases where model might have different number of metrics
    if len(results) >= 5:
        precision = results[2]
        recall = results[3]
        auc = results[4]
    elif len(results) >= 3:
        precision = results[2]
        recall = results[2]  # Same as precision if only 3 metrics
        auc = results[2]
    else:
        precision = accuracy
        recall = accuracy
        auc = accuracy


    print(" TEST SET RESULTS")

    print(f"  Loss:      {loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Calculate F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"  F1-Score:  {f1_score:.4f}")
    else:
        f1_score = 0.0
        print(f"  F1-Score:  N/A")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f1_score": f1_score,
    }


# DETAILED CLASSIFICATION REPORT


def generate_classification_report(model, test_generator):
    """Generate detailed classification report and confusion matrix."""

    print(" DETAILED CLASSIFICATION REPORT")


    test_generator.reset()

    # Get predictions
    print("\n Generating predictions...")
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes

    # Get class names
    class_names = list(test_generator.class_indices.keys())

    # Classification report

    print(" CLASSIFICATION REPORT")

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 12},
    )
    plt.title(
        "Confusion Matrix - Fruit Freshness Classification", fontsize=16, weight="bold"
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("\n Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

    # Per-class accuracy

    print(" PER-CLASS ACCURACY")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(
            f"  {class_name:20s}: {class_correct:4d}/{class_total:4d} = {class_accuracy:.4f} ({class_accuracy*100:6.2f}%)"
        )

    return y_true, y_pred, y_pred_proba, class_names, cm
