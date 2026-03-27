import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle

from utils.faced_meta import build_faced_meta_maps, lmdb_key_to_subject_meta

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            return_keys: bool = False,
            return_metadata: bool = False,
            meta_maps=None,
    ):
        super(CustomDataset, self).__init__()
        self.return_keys = return_keys
        self.return_metadata = return_metadata
        self.meta_maps = meta_maps or {}
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
        if self.return_keys or self.return_metadata:
            out = [data / 100, label]
            if self.return_keys:
                out.append(kstr)
            if self.return_metadata:
                out.append(lmdb_key_to_subject_meta(kstr, self.meta_maps))
            return tuple(out)
        return data / 100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        if self.return_keys or self.return_metadata:
            ptr = 2
            keys = None
            batch_meta = None
            if self.return_keys:
                keys = [x[ptr] for x in batch]
                ptr += 1
            if self.return_metadata:
                metas = [x[ptr] for x in batch]
                bsz = len(metas)
                subj_summary = [m.get('subject_summary') for m in metas]
                valid_summary = [s for s in subj_summary if s is not None]
                summary_tensor = None
                if len(valid_summary) == bsz:
                    try:
                        summary_tensor = torch.as_tensor(np.array(subj_summary), dtype=torch.float32)
                    except Exception:
                        summary_tensor = None
                batch_meta = {
                    'subject_id': torch.tensor([m.get('subject_id', 0) for m in metas], dtype=torch.long),
                    'dataset_id': torch.tensor([m.get('dataset_id', 0) for m in metas], dtype=torch.long),
                    'cohort_id': torch.tensor([m['cohort_id'] for m in metas], dtype=torch.long),
                    'sample_rate_group_id': torch.tensor([m['sample_rate_group_id'] for m in metas], dtype=torch.long),
                    'age_bucket_id': torch.tensor([m['age_bucket_id'] for m in metas], dtype=torch.long),
                    'segment_bucket_id': torch.tensor([m['segment_bucket_id'] for m in metas], dtype=torch.long),
                    'channel_count': torch.full((bsz,), int(x_data.shape[1]), dtype=torch.float32),
                }
                if summary_tensor is not None:
                    batch_meta['subject_summary'] = summary_tensor
            if keys is not None and batch_meta is not None:
                return to_tensor(x_data), to_tensor(y_label).long(), keys, batch_meta
            if keys is not None:
                return to_tensor(x_data), to_tensor(y_label).long(), keys
            return to_tensor(x_data), to_tensor(y_label).long(), batch_meta
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        rk = getattr(self.params, "return_sample_keys", False)
        return_metadata = bool(
            getattr(self.params, 'subject_adapter', False)
            or getattr(self.params, 'eeg_channel_context', False)
            or getattr(self.params, 'continual_mode', 'off') != 'off'
            or (
                getattr(self.params, 'moe', False)
                and getattr(self.params, 'moe_route_mode', '') == 'typed_capacity_domain'
            )
        )
        meta_maps = build_faced_meta_maps(
            self.datasets_dir,
            getattr(self.params, 'faced_meta_csv', ''),
            getattr(self.params, 'subject_summary_file', ''),
        ) if return_metadata else {}

        train_set = CustomDataset(
            self.datasets_dir,
            mode='train',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
        )
        val_set = CustomDataset(
            self.datasets_dir,
            mode='val',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
        )
        test_set = CustomDataset(
            self.datasets_dir,
            mode='test',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
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
