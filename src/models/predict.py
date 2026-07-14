from pathlib import Path
import json
import yaml
import numpy as np
import easyocr
import tensorflow as tf

FAKE_NOTE_KEYWORDS = [
    "SPECIMEN", "COPY", "PLAY MONEY", "PROP",
    "PROP MONEY", "MOTION PICTURE",
    "FOR MOTION PICTURE USE", "MOVIE MONEY",
    "REPLICA", "SAMPLE", "FACSIMILE", "IMITATION",
    "NOVELTY", "TOY", "FUN", "FUNNY", "CHILDREN",
    "KIDS", "EDUCATIONAL", "NOT LEGAL TENDER",
    "MONOPOLY", "MANORANJAN", "ENTERTAINMENT",
    "COUPON", "VOUCHER", "DEMO", "TEST",
    "PRACTICE", "TRAINING", "GIFT",
    "PROMOTIONAL", "ADVERTISEMENT",
]

REAL_NOTE_KEYWORDS = [
    "RESERVE", "BANK", "INDIA",
    "RESERVE BANK", "RESERVE BANK OF INDIA", "RBI",
    "भारत", "भारतीय", "रिज़र्व", "बैंक",
    "₹", "RUPEES", "RUPEE", "GOVERNOR",
    "GUARANTEED", "PROMISE", "PAY", "MAHATMA",
    "GANDHI", "सत्यमेव", "JAYATE", "SATYAMEVA",
    "BHARAT",
]

REAL_KEYWORD_THRESHOLD = 3

_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(
            ["en", "hi"], gpu=False
        )
    return _ocr_reader


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


def ocr_detect(image_input):
    """Run OCR on image, return extracted text."""
    reader = get_ocr_reader()
    if isinstance(image_input, bytes):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(image_input))
        img = np.array(img)
    else:
        img = str(image_input)
    results = reader.readtext(img, detail=0)
    return " ".join(results).upper()


def check_fake_keywords(text):
    """Return True if any fake keyword found."""
    for kw in FAKE_NOTE_KEYWORDS:
        if kw.upper() in text:
            return True, kw
    return False, None


def check_real_keywords(text):
    """Return count of real keywords found."""
    count = 0
    for kw in REAL_NOTE_KEYWORDS:
        if kw.upper() in text:
            count += 1
    return count


def predict(model, image, image_input=None):
    """Full prediction pipeline: OCR check then model."""
    # --- Step 1: OCR keyword check ---
    if image_input is not None:
        text = ocr_detect(image_input)
        is_fake, kw = check_fake_keywords(text)
        if is_fake:
            return 0, 1.0, f"OCR: fake keyword '{kw}' found"

        real_count = check_real_keywords(text)
        if real_count <= REAL_KEYWORD_THRESHOLD:
            return 0, 1.0, (
                f"OCR: only {real_count} real keywords "
                f"(need > {REAL_KEYWORD_THRESHOLD})"
            )

    # --- Step 2: Model prediction ---
    auth_pred, _ = model.predict(image, verbose=0)
    auth_idx = int(auth_pred[0] > 0.5)
    if auth_idx == 1:
        auth_confidence = float(auth_pred[0])
    else:
        auth_confidence = float(1 - auth_pred[0])

    return auth_idx, auth_confidence, "Model prediction"


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
    auth_idx, auth_conf, source = predict(model, image, image_path)

    result = {
        "image": str(image_path),
        "authenticity": AUTH_CLASSES[auth_idx],
        "confidence": round(auth_conf, 4),
        "source": source,
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
