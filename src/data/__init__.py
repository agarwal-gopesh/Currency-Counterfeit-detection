import tensorflow as tf
from pathlib import Path

project_root = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path.cwd()
raw_path = project_root / "data" / "raw"

bad_files = []

for img_path in raw_path.rglob("*.*"):
    try:
        img_bytes = tf.io.read_file(str(img_path))
        tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    except Exception as e:
        bad_files.append(str(img_path))
        print(f"BAD FILE: {img_path} — {e}")

print(f"\nTotal bad files: {len(bad_files)}")