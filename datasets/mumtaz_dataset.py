import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            raw_keys = txn.get(b'__keys__')
        if raw_keys is None:
            raise KeyError(f"Mumtaz2016 LMDB missing '__keys__' in {data_dir}")

        split_index = pickle.loads(raw_keys)
        if mode not in split_index:
            raise KeyError(f"Mumtaz2016 LMDB missing split {mode!r}; available: {list(split_index.keys())}")
        self.keys = split_index[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        enc_key = key.encode() if isinstance(key, str) else key
        with self.db.begin(write=False) as txn:
            raw = txn.get(enc_key)
        if raw is None:
            raise KeyError(f"Mumtaz2016 LMDB key not found: {key!r}")
        pair = pickle.loads(raw)
        data = pair['sample']
        label = pair['label']
        return data / 100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')

        num_workers = int(getattr(self.params, 'num_workers', 0))
        pin_memory = bool(getattr(self.params, 'pin_memory', False))
        persistent_workers = bool(getattr(self.params, 'persistent_workers', False) and num_workers > 0)

        print(
            f"[Mumtaz2016 split] train={len(train_set)} val={len(val_set)} "
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
