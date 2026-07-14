from pathlib import Path
import json
import yaml
import tensorflow as tf


def load_params():
    root = Path(__file__).resolve().parents[2]
    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params, root


def load_saved_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_input, img_size=224):
    # --- Streamlit: image_input is raw bytes ---
    if isinstance(image_input, bytes):
        img = tf.io.decode_image(
            image_input, channels=3, expand_animations=False
        )
    # --- DVC pipeline: image_input is a file path ---
    else:
        img = tf.io.read_file(str(image_input))
        img = tf.image.decode_image(
            img, channels=3, expand_animations=False
        )

    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img


def predict(model, image):
    auth_pred, _ = model.predict(image, verbose=0)

    auth_idx = int(auth_pred[0] > 0.5)

    if auth_idx == 1:
        auth_confidence = float(auth_pred[0])
    else:
        auth_confidence = float(1 - auth_pred[0])

    return auth_idx, auth_confidence


def main():
    params, project_root = load_params()

    model_path = (
        project_root / params["predict"]["model_path"]
    )
    output_dir = (
        project_root / params["predict"]["output_dir"]
    )
    image_path = (
        project_root / params["predict"]["image_path"]
    )
    img_size = params["train"]["img_size"]

    AUTH_CLASSES = ["fake", "real"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_saved_model(model_path)
    image = preprocess_image(image_path, img_size)
    auth_idx, auth_conf = predict(model, image)

    result = {
        "image": str(image_path),
        "authenticity": AUTH_CLASSES[auth_idx],
        "confidence": round(auth_conf, 4),
    }

    print("\n===== PREDICTION =====")
    auth_str = result["authenticity"]
    conf_str = f"{result['confidence']:.1%}"
    print(f"Authenticity : {auth_str} ({conf_str})")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{image_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    main()
