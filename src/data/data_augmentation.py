import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ---- Paths ----
project_root = Path(__file__).resolve().parents[2]
train_csv_path = project_root / "data" / "processed" / "train.csv"
test_csv_path = project_root / "data" / "processed" / "test.csv"

# ---- Load CSVs ----
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# ---- Add full path ----
train_df["full_path"] = str(project_root) + "/data/" + train_df["image_path"]
test_df["full_path"] = str(project_root) + "/data/" + test_df["image_path"]

# ---- Label Encoding (differentiate between authenticity and denomination) ----
auth_encoder = LabelEncoder()
denom_encoder = LabelEncoder()

train_df["authenticity_encoded"] = auth_encoder.fit_transform(train_df["authenticity"])
train_df["denomination_encoded"] = denom_encoder.fit_transform(train_df["denomination"])

test_df["authenticity_encoded"] = auth_encoder.transform(test_df["authenticity"])
test_df["denomination_encoded"] = denom_encoder.transform(test_df["denomination"])

print("Authenticity classes:", auth_encoder.classes_)
print("Denomination classes:", denom_encoder.classes_)

# ---- Constants ----
IMG_SIZE = 224
BATCH_SIZE = 32

# ---- Image loading function ----
def load_image(path, auth_label, denom_label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, {"authenticity_output": auth_label, "denomination_output": denom_label}

# ---- Augmentation function (applicable only for training) ----
def augment(img, labels):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, labels

# ---- Build tf.data pipeline: TRAIN (with augmentation) ----
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

# ---- Build tf.data pipeline: TEST (augmentation NOT applicable) ----
test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["full_path"].values,
    test_df["authenticity_encoded"].values,
    test_df["denomination_encoded"].values
))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ---- Quick sanity check ----
if __name__ == "__main__":
    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Authenticity labels:", labels["authenticity_output"].numpy())
        print("Denomination labels:", labels["denomination_output"].numpy())