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

train_df["label"] = auth_encoder.fit_transform(train_df["authenticity"])
test_df["label"] = auth_encoder.transform(test_df["authenticity"])

print("Classes :", auth_encoder.classes_)

IMG_SIZE = 224
BATCH_SIZE = 32

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

def augment(img, label):
    img = data_augmentation(img)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((
    train_df["full_path"].values,
    train_df["label"].values
))

train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["full_path"].values,
    test_df["label"].values
))

test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

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
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="authenticity_output"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

if __name__ == "__main__":

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    print("\n===== PHASE 1 =====\n")

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=10,
        callbacks=[early_stop]
    )

    print("\n===== PHASE 2 =====\n")

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=7,
        callbacks=[early_stop]
    )

    print("\n===== EVALUATION =====\n")

    results = model.evaluate(test_ds)
    print(results)

    model_dir = project_root / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / "authenticity_model.keras")    # in authenticity.py

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=auth_encoder.classes_,
        yticklabels=auth_encoder.classes_
    )

    plt.title("Authenticity Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        project_root /
        "reports" /
        "figures" /
        "confusion_matrix_authenticity.png"
    )

    plt.show()

    print(classification_report(
        y_true,
        y_pred,
        target_names=auth_encoder.classes_
    ))