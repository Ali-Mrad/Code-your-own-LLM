import urllib.request, zipfile, os, pathlib, pandas as pd

URL  = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILE = "SMSSpamCollection"          


def _download_and_extract(zip_path: pathlib.Path, extract_to: pathlib.Path):
    """Download the ZIP and extract the raw file."""
    print("downloading SMS Spam Collection …")
    with urllib.request.urlopen(URL) as r:
        zip_path.write_bytes(r.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(RAW_FILE, path=extract_to)
    print("dataset extracted")


def _random_split(df: pd.DataFrame, train_frac: float, val_frac: float, seed: int):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end   = train_end + int(len(df) * val_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def prepare_dataset(
    root: str | pathlib.Path = "data/sms_spam",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 123,
):

    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path  = root / "sms_spam_collection.zip"
    raw_path  = root / "SMSSpamCollection.tsv"              
    csv_train = root / "train.csv"
    csv_val   = root / "validation.csv"
    csv_test  = root / "test.csv"

    if not raw_path.exists():
        _download_and_extract(zip_path, root)
        os.rename(root / RAW_FILE, raw_path)

    df = pd.read_csv(raw_path, sep="\t", header=None, names=["Label", "Text"])

    n_spam = df[df.Label == "spam"].shape[0]
    ham_balanced = df[df.Label == "ham"].sample(n_spam, random_state=seed)
    balanced = pd.concat([ham_balanced, df[df.Label == "spam"]], ignore_index=True)
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})


    train_df, val_df, test_df = _random_split(balanced, train_frac, val_frac, seed)
    train_df.to_csv(csv_train, index=False)
    val_df.to_csv(csv_val,   index=False)
    test_df.to_csv(csv_test, index=False)

    print(" CSVs saved to", root.resolve())
