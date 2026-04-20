import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import pickle


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            files,
    ):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.files = files

    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.data_dir, file), "rb") as f:
            data_dict = pickle.load(f)
        data = data_dict['signal']
        label = int(data_dict['label'][0]-1)
        data = data.reshape(16, 5, 200)
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        processed_root = self._resolve_processed_root(self.datasets_dir)
        train_dir = os.path.join(processed_root, "processed_train")
        val_dir = os.path.join(processed_root, "processed_eval")
        test_dir = os.path.join(processed_root, "processed_test")

        train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pkl')])
        val_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.pkl')])
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.pkl')])

        train_set = CustomDataset(train_dir, train_files)
        val_set = CustomDataset(val_dir, val_files)
        test_set = CustomDataset(test_dir, test_files)

        num_workers = int(getattr(self.params, 'num_workers', 0))
        pin_memory = bool(getattr(self.params, 'pin_memory', False))
        persistent_workers = bool(getattr(self.params, 'persistent_workers', False) and num_workers > 0)

        print(
            f"[TUEV split] root={processed_root} train={len(train_set)} val={len(val_set)} "
            f"test={len(test_set)} total={len(train_set) + len(val_set) + len(test_set)}"
        )

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            ),
        }
        return data_loader

    @staticmethod
    def _resolve_processed_root(datasets_dir):
        candidates = [
            datasets_dir,
            os.path.join(datasets_dir, "processed"),
        ]
        for candidate in candidates:
            required = [
                os.path.join(candidate, "processed_train"),
                os.path.join(candidate, "processed_eval"),
                os.path.join(candidate, "processed_test"),
            ]
            if all(os.path.isdir(path) for path in required):
                return candidate
        raise FileNotFoundError(
            "TUEV datasets_dir must contain processed_train/processed_eval/processed_test "
            f"directly or under a nested 'processed' folder. Got: {datasets_dir}"
        )
