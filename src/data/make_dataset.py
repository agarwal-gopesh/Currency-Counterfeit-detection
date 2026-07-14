from pathlib import Path
import yaml
import pandas as pd
import sys
# import mlflow
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def load_params():
    root = Path(__file__).resolve().parents[2]
    with open(root / "params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["make_dataset"], root


def scan_raw_data(raw_data_path):
    rows = []
    for authenticity_dir in raw_data_path.iterdir():
        if not authenticity_dir.is_dir():
            continue
        authenticity = authenticity_dir.name

        for denom_dir in authenticity_dir.iterdir():
            if not denom_dir.is_dir():
                continue
            denomination = denom_dir.name

            for img_path in denom_dir.glob("*.*"):
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                rows.append({
                    "image_path": str(img_path.relative_to(raw_data_path.parents[0])),
                    "authenticity": authenticity,
                    "denomination": denomination,
                    "combined_label": f"{authenticity}_{denomination}"
                })

    df = pd.DataFrame(rows)
    return df


def split_data(df, test_split, seed):
    train, test = train_test_split(
        df,
        test_size=test_split,
        random_state=seed,
        stratify=df["combined_label"]
    )
    return train, test


def save_data(train, test, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


def main():
    params, root = load_params()
    input_dir = sys.argv[1]
    raw_data_path = root / input_dir

    df = scan_raw_data(raw_data_path)

    # mlflow.set_experiment("make_dataset")

    # with mlflow.start_run():
    #     mlflow.log_param("test_split", params["test_split"])
    #     mlflow.log_param("seed", params["seed"])
    #     mlflow.log_param("total_images", len(df))

    train, test = split_data(df, params["test_split"], params["seed"])

    #     mlflow.log_metric("train_size", len(train))
    #     mlflow.log_metric("test_size", len(test))

    save_data(train, test, root / "data" / "processed")


if __name__ == "__main__":
    main()