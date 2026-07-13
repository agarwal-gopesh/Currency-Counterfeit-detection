from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


def load_params():
    root = Path(__file__).resolve().parents[2]
    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"], root


def build_dataset(train_df, test_df, img_size, batch_size):
    auth_encoder = LabelEncoder()
    denom_encoder = LabelEncoder()

    train_df["auth_encoded"] = auth_encoder.fit_transform(train_df["authenticity"])
    train_df["denom_encoded"] = denom_encoder.fit_transform(train_df["denomination"])

    test_df["auth_encoded"] = auth_encoder.transform(test_df["authenticity"])
    test_df["denom_encoded"] = denom_encoder.transform(test_df["denomination"])

    num_denom = len(denom_encoder.classes_)

    def load_image(path, auth_label, denom_label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, {"authenticity_output": auth_label, "denomination_output": denom_label}

    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def augment(img, labels):
        return augmentation(img), labels

    train_ds = tf.data.Dataset.from_tensor_slices((
        train_df["full_path"].values,
        train_df["auth_encoded"].values,
        train_df["denom_encoded"].values,
    ))
    train_ds = train_ds.map(load_image, num_parallel_calls=2)
    train_ds = train_ds.map(augment, num_parallel_calls=2)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(1)

    test_ds = tf.data.Dataset.from_tensor_slices((
        test_df["full_path"].values,
        test_df["auth_encoded"].values,
        test_df["denom_encoded"].values,
    ))
    test_ds = test_ds.map(load_image, num_parallel_calls=2)
    test_ds = test_ds.batch(batch_size).prefetch(1)

    return train_ds, test_ds, auth_encoder, denom_encoder, num_denom


def build_model(img_size, num_denom):
    base_model = MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    auth_branch = layers.Dense(64, activation="relu")(x)
    auth_output = layers.Dense(1, activation="sigmoid", name="authenticity_output")(auth_branch)

    denom_branch = layers.Dense(64, activation="relu")(x)
    denom_branch = layers.Dropout(0.35)(denom_branch)
    denom_output = layers.Dense(num_denom, activation="softmax", name="denomination_output")(denom_branch)

    model = Model(inputs=inputs, outputs=[auth_output, denom_output])
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "authenticity_output": "binary_crossentropy",
            "denomination_output": "sparse_categorical_crossentropy",
        },
        loss_weights={"authenticity_output": 1.0, "denomination_output": 1.0},
        metrics={
            "authenticity_output": "accuracy",
            "denomination_output": "accuracy",
        },
    )


def evaluate_and_plot(model, test_ds, auth_encoder, denom_encoder, figures_dir):
    y_true_auth, y_true_denom, y_pred_auth, y_pred_denom = [], [], [], []

    for images, labels in test_ds:
        auth_pred, denom_pred = model.predict(images, verbose=0)
        y_true_auth.extend(labels["authenticity_output"].numpy())
        y_true_denom.extend(labels["denomination_output"].numpy())
        y_pred_auth.extend((auth_pred > 0.5).astype(int).flatten())
        y_pred_denom.extend(np.argmax(denom_pred, axis=1))

    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_auth = confusion_matrix(y_true_auth, y_pred_auth)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_auth, annot=True, fmt="d", cmap="Blues",
                xticklabels=auth_encoder.classes_, yticklabels=auth_encoder.classes_)
    plt.title("Authenticity Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_authenticity.png")
    plt.close()

    cm_denom = confusion_matrix(y_true_denom, y_pred_denom)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_denom, annot=True, fmt="d", cmap="Greens",
                xticklabels=denom_encoder.classes_, yticklabels=denom_encoder.classes_)
    plt.title("Denomination Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_denomination.png")
    plt.close()

    print("\nAuthenticity Classification Report:")
    print(classification_report(y_true_auth, y_pred_auth, target_names=auth_encoder.classes_))
    print("Denomination Classification Report:")
    print(classification_report(y_true_denom, y_pred_denom, target_names=[str(c) for c in denom_encoder.classes_]))


def main():
    print("Starting build_dataset...")
    params, root = load_params()

    img_size = params["img_size"]
    batch_size = params["batch_size"]
    phase1_epochs = params["phase1_epochs"]
    phase1_lr = params["phase1_lr"]
    phase1_patience = params["phase1_patience"]
    phase2_epochs = params["phase2_epochs"]
    phase2_lr = params["phase2_lr"]
    phase2_patience = params["phase2_patience"]
    phase2_unfreeze_last = params["phase2_unfreeze_last"]

    train_df = pd.read_csv(root / "data" / "processed" / "train.csv")
    test_df = pd.read_csv(root / "data" / "processed" / "test.csv")

    data_dir = root / "data"
    train_df["full_path"] = str(data_dir) + "/" + train_df["image_path"]
    test_df["full_path"] = str(data_dir) + "/" + test_df["image_path"]

    train_ds, test_ds, auth_encoder, denom_encoder, num_denom = build_dataset(
        train_df, test_df, img_size, batch_size
    )
    print("build_dataset DONE")

    print("Authenticity classes:", auth_encoder.classes_)
    print("Denomination classes:", denom_encoder.classes_)

    model, base_model = build_model(img_size, num_denom)
    model.summary()

    mlflow.set_experiment("train_model")

    with mlflow.start_run():
        mlflow.log_params({
            "img_size": img_size,
            "batch_size": batch_size,
            "phase1_epochs": phase1_epochs,
            "phase1_lr": phase1_lr,
            "phase1_patience": phase1_patience,
            "phase2_epochs": phase2_epochs,
            "phase2_lr": phase2_lr,
            "phase2_patience": phase2_patience,
            "phase2_unfreeze_last": phase2_unfreeze_last,
        })
        mlflow.log_param("num_denom_classes", num_denom)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))

        # ---- Phase 1: Frozen backbone ----
        print("\n===== PHASE 1: Frozen backbone =====\n")
        compile_model(model, phase1_lr)

        history_p1 = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=phase1_epochs,
            verbose=2,
            callbacks=[EarlyStopping(
                monitor="val_denomination_output_loss", mode="min",
                patience=phase1_patience, restore_best_weights=True, verbose=1,
            )],
        )

        best_p1_val_auth = max(history_p1.history["val_authenticity_output_accuracy"])
        best_p1_val_denom = max(history_p1.history["val_denomination_output_accuracy"])
        mlflow.log_metric("phase1_val_auth_accuracy", best_p1_val_auth)
        mlflow.log_metric("phase1_val_denom_accuracy", best_p1_val_denom)

        # ---- Phase 2: Fine-tuning ----
        print("\n===== PHASE 2: Fine-tuning =====\n")
        base_model.trainable = True
        for layer in base_model.layers[:-phase2_unfreeze_last]:
            layer.trainable = False

        compile_model(model, phase2_lr)

        history_p2 = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=phase2_epochs,
            verbose=2,
            callbacks=[EarlyStopping(
                monitor="val_denomination_output_loss", mode="min",
                patience=phase2_patience, restore_best_weights=True, verbose=1,
            )],
        )

        best_p2_val_auth = max(history_p2.history["val_authenticity_output_accuracy"])
        best_p2_val_denom = max(history_p2.history["val_denomination_output_accuracy"])
        mlflow.log_metric("phase2_val_auth_accuracy", best_p2_val_auth)
        mlflow.log_metric("phase2_val_denom_accuracy", best_p2_val_denom)

        # ---- Final evaluation ----
        print("\n===== FINAL EVALUATION =====\n")
        results = model.evaluate(test_ds)
        print(results)

        # ---- Save model ----
        model_dir = root / "artifacts" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / "multi_output_model.keras")
        # mlflow.log_artifact(str(model_dir / "multi_output_model.keras"))

        # ---- Confusion matrices ----
        figures_dir = root / "reports" / "figures"
        evaluate_and_plot(model, test_ds, auth_encoder, denom_encoder, figures_dir)


if __name__ == "__main__":
    main()
