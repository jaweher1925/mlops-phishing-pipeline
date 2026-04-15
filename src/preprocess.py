import pandas as pd
import re
import os
import yaml
from sklearn.model_selection import train_test_split

# Load params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_path"]
PROCESSED_PATH = params["data"]["processed_path"]
TEST_SIZE = params["data"]["test_size"]
RANDOM_STATE = params["data"]["random_state"]

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # remove special chars
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text

def preprocess():
    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH)

    print(f"Original shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Drop garbage class (label == 2)
    df = df[df["label"] != 2].copy()
    print(f"After removing garbage: {df.shape}")

    # Combine subject + body into one text column
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = df["subject"] + " " + df["body"]
    df["text"] = df["text"].apply(clean_text)

    # Keep only what we need
    df = df[["text", "label"]].dropna()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # Save
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df.to_csv(f"{PROCESSED_PATH}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_PATH}/test.csv", index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("Preprocessing done!")

if __name__ == "__main__":
    preprocess()