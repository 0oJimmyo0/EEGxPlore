import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle

from utils.faced_meta import build_faced_domain_maps, lmdb_key_to_domain_ids

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            return_keys: bool = False,
            return_domain_ids: bool = False,
            domain_maps=None,
    ):
        super(CustomDataset, self).__init__()
        self.return_keys = return_keys
        self.return_domain_ids = return_domain_ids
        self.domain_maps = domain_maps or {}
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        enc = key.encode() if isinstance(key, str) else key
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(enc))
        data = pair['sample']
        label = pair['label']
        kstr = key.decode() if isinstance(key, bytes) else str(key)
        if self.return_keys or self.return_domain_ids:
            out = [data / 100, label]
            if self.return_keys:
                out.append(kstr)
            if self.return_domain_ids:
                out.append(lmdb_key_to_domain_ids(kstr, self.domain_maps))
            return tuple(out)
        return data / 100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        if self.return_keys or self.return_domain_ids:
            ptr = 2
            keys = None
            domain_meta = None
            if self.return_keys:
                keys = [x[ptr] for x in batch]
                ptr += 1
            if self.return_domain_ids:
                metas = [x[ptr] for x in batch]
                domain_meta = {
                    'cohort_id': torch.tensor([m['cohort_id'] for m in metas], dtype=torch.long),
                    'sample_rate_group_id': torch.tensor([m['sample_rate_group_id'] for m in metas], dtype=torch.long),
                    'age_bucket_id': torch.tensor([m['age_bucket_id'] for m in metas], dtype=torch.long),
                    'segment_bucket_id': torch.tensor([m['segment_bucket_id'] for m in metas], dtype=torch.long),
                }
            if keys is not None and domain_meta is not None:
                return to_tensor(x_data), to_tensor(y_label).long(), keys, domain_meta
            if keys is not None:
                return to_tensor(x_data), to_tensor(y_label).long(), keys
            return to_tensor(x_data), to_tensor(y_label).long(), domain_meta
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        rk = getattr(self.params, "return_sample_keys", False)
        route_mode = getattr(self.params, 'moe_route_mode', '')
        return_domain_ids = bool(
            getattr(self.params, 'moe', False)
            and route_mode == 'typed_capacity_domain'
        )
        domain_maps = build_faced_domain_maps(getattr(self.params, 'faced_meta_csv', '')) if return_domain_ids else {}

        train_set = CustomDataset(
            self.datasets_dir,
            mode='train',
            return_keys=rk,
            return_domain_ids=return_domain_ids,
            domain_maps=domain_maps,
        )
        val_set = CustomDataset(
            self.datasets_dir,
            mode='val',
            return_keys=rk,
            return_domain_ids=return_domain_ids,
            domain_maps=domain_maps,
        )
        test_set = CustomDataset(
            self.datasets_dir,
            mode='test',
            return_keys=rk,
            return_domain_ids=return_domain_ids,
            domain_maps=domain_maps,
        )
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
