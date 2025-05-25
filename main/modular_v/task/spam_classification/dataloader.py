from torch.utils.data import Dataset, DataLoader
import pathlib, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

class SpamDataset(Dataset):
  

    def __init__(
        self,
        csv_file: str | pathlib.Path,
        tokenizer,
        max_length: int | None = None,
        pad_token_id: int = 50_256,  # GPT-2 <|endoftext|>
    ):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(txt) for txt in self.data["Text"]]

        self.max_length = max_length or max(len(seq) for seq in self.encoded_texts)

        self.encoded_texts = [
            seq[: self.max_length] + [pad_token_id] * (self.max_length - len(seq))
            for seq in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.long),
        )

def get_dataloaders(
    tokenizer,
    data_root: str | pathlib.Path = "data/sms_spam",
    batch_size: int = 8,
    num_workers: int = 0,
    pad_token_id: int = 50_256,
):
   
    root = pathlib.Path(data_root)
    csv_train = root / "train.csv"
    csv_val   = root / "validation.csv"
    csv_test  = root / "test.csv"

    # create train first to compute max_length
    train_ds = SpamDataset(csv_train, tokenizer, max_length=None, pad_token_id=pad_token_id)
    max_len  = train_ds.max_length

    val_ds  = SpamDataset(csv_val,  tokenizer, max_length=max_len, pad_token_id=pad_token_id)
    test_ds = SpamDataset(csv_test, tokenizer, max_length=max_len, pad_token_id=pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, max_len

