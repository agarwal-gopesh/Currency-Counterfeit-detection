import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
train_csv_path = project_root / "data" / "processed" / "train.csv"
test_csv_path = project_root / "data" / "processed" / "test.csv"

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

train_df["full_path"] = str(project_root / "data") + "/" + train_df["image_path"]
test_df["full_path"] = str(project_root / "data") + "/" + test_df["image_path"]

auth_encoder = LabelEncoder()
denom_encoder = LabelEncoder()

train_df["authenticity_encoded"] = auth_encoder.fit_transform(train_df["authenticity"])
train_df["denomination_encoded"] = denom_encoder.fit_transform(train_df["denomination"])

test_df["authenticity_encoded"] = auth_encoder.transform(test_df["authenticity"])
test_df["denomination_encoded"] = denom_encoder.transform(test_df["denomination"])

NUM_DENOMINATIONS = len(denom_encoder.classes_)

print("Authenticity classes:", auth_encoder.classes_)
print("Denomination classes:", denom_encoder.classes_)

IMG_SIZE = 224
BATCH_SIZE = 32

def load_image(path, auth_label, denom_label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, {"authenticity_output": auth_label, "denomination_output": denom_label}

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

def augment(img, labels):
    img = data_augmentation(img)
    return img, labels

train_ds = tf.data.Dataset.from_tensor_slices((
    train_df["full_path"].values,
    train_df["authenticity_encoded"].values,
    train_df["denomination_encoded"].values
))
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["full_path"].values,
    test_df["authenticity_encoded"].values,
    test_df["denomination_encoded"].values
))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ---- Build Model with Transfer Learning ----
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

auth_branch = layers.Dense(64, activation="relu")(x)
auth_output = layers.Dense(1, activation="sigmoid", name="authenticity_output")(auth_branch)

denom_branch = layers.Dense(64, activation="relu")(x)
denom_branch = layers.Dropout(0.35)(denom_branch)  # extra dropout, denomination overfit kar raha tha
denom_output = layers.Dense(NUM_DENOMINATIONS, activation="softmax", name="denomination_output")(denom_branch)

model = Model(inputs=inputs, outputs=[auth_output, denom_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "authenticity_output": "binary_crossentropy",
        "denomination_output": "sparse_categorical_crossentropy"
    },
    loss_weights={
        "authenticity_output": 1.0,
        "denomination_output": 1.0
    },
    metrics={
        "authenticity_output": "accuracy",
        "denomination_output": "accuracy"
    }
)

model.summary()

if __name__ == "__main__":
    # ---- Phase 1: Frozen backbone ----
    early_stop_phase1 = EarlyStopping(
    monitor="val_denomination_output_loss",
    mode="min",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

    print("\n===== PHASE 1: Training with frozen backbone =====\n")
    history_phase1 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=[early_stop_phase1]
    )

    # ---- Phase 2: Fine-tuning (unfreeze top layers of backbone) ----
    print("\n===== PHASE 2: Fine-tuning top layers of backbone =====\n")

    base_model.trainable = True

    # sirf backbone ke last 30 layers unfreeze karo, baaki freeze rehne do
    # (backbone ke early layers generic features seekhte hain - edges, textures -
    #  unhe touch karna zaroori nahi; sirf later layers currency-specific banao)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # bahut chhota LR, warna weights bigad jayenge
        loss={
            "authenticity_output": "binary_crossentropy",
            "denomination_output": "sparse_categorical_crossentropy"
        },
        loss_weights={
            "authenticity_output": 1.0,
            "denomination_output": 1.0
        },
        metrics={
            "authenticity_output": "accuracy",
            "denomination_output": "accuracy"
        }
    )

    early_stop_phase2 = EarlyStopping(
    monitor="val_denomination_output_loss",
    mode="min",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

    history_phase2 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=7,
        callbacks=[early_stop_phase2]
    )

    # ---- Final Evaluation ----
    print("\n===== FINAL EVALUATION =====\n")
    results = model.evaluate(test_ds)
    print(results)

    # ---- Confusion Matrix ----
    print("\n===== GENERATING CONFUSION MATRIX =====\n")

    y_true_auth = []
    y_true_denom = []
    y_pred_auth = []
    y_pred_denom = []

    for images, labels in test_ds:
        auth_pred, denom_pred = model.predict(images, verbose=0)

        y_true_auth.extend(labels["authenticity_output"].numpy())
        y_true_denom.extend(labels["denomination_output"].numpy())

        y_pred_auth.extend((auth_pred > 0.5).astype(int).flatten())
        y_pred_denom.extend(np.argmax(denom_pred, axis=1))

    # ---- Authenticity Confusion Matrix ----
    cm_auth = confusion_matrix(y_true_auth, y_pred_auth)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_auth, annot=True, fmt="d", cmap="Blues",
                xticklabels=auth_encoder.classes_,
                yticklabels=auth_encoder.classes_)
    plt.title("Authenticity Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(project_root / "reports" / "figures" / "confusion_matrix_authenticity.png")
    plt.show()

    print("\nAuthenticity Classification Report:")
    print(classification_report(y_true_auth, y_pred_auth, target_names=auth_encoder.classes_))

    # ---- Denomination Confusion Matrix ----
    cm_denom = confusion_matrix(y_true_denom, y_pred_denom)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_denom, annot=True, fmt="d", cmap="Greens",
                xticklabels=denom_encoder.classes_,
                yticklabels=denom_encoder.classes_)
    plt.title("Denomination Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(project_root / "reports" / "figures" / "confusion_matrix_denomination.png")
    plt.show()

    print("\nDenomination Classification Report:")
    print(classification_report(y_true_denom, y_pred_denom, target_names=[str(c) for c in denom_encoder.classes_]))