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
            use_subject_summary: bool = False,
    ):
        super(CustomDataset, self).__init__()
        self.return_keys = return_keys
        self.return_metadata = return_metadata
        self.meta_maps = meta_maps or {}
        self.use_subject_summary = bool(use_subject_summary)
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
                batch_meta = {
                    'subject_id': torch.tensor([m.get('subject_id', 0) for m in metas], dtype=torch.long),
                    'dataset_id': torch.tensor([m.get('dataset_id', 0) for m in metas], dtype=torch.long),
                    'cohort_id': torch.tensor([m['cohort_id'] for m in metas], dtype=torch.long),
                    'sample_rate_group_id': torch.tensor([m['sample_rate_group_id'] for m in metas], dtype=torch.long),
                    'age_bucket_id': torch.tensor([m['age_bucket_id'] for m in metas], dtype=torch.long),
                    'segment_bucket_id': torch.tensor([m['segment_bucket_id'] for m in metas], dtype=torch.long),
                    'channel_count': torch.full((bsz,), int(x_data.shape[1]), dtype=torch.float32),
                }
                if self.use_subject_summary:
                    summary_rows = []
                    for i, m in enumerate(metas):
                        s = m.get('subject_summary')
                        if s is None:
                            raise ValueError(
                                "use_subject_summary=True but subject_summary is missing for "
                                f"sample index {i} in this batch. Provide a complete "
                                "--subject_summary_file or disable --use_subject_summary."
                            )
                        arr = np.asarray(s, dtype=np.float32).reshape(-1)
                        if arr.size == 0:
                            raise ValueError("subject_summary vector is empty; expected non-empty 1D vector.")
                        summary_rows.append(arr)
                    dim0 = int(summary_rows[0].shape[0])
                    for i, arr in enumerate(summary_rows):
                        if int(arr.shape[0]) != dim0:
                            raise ValueError(
                                "subject_summary vectors must have a consistent size within batch; "
                                f"got {arr.shape[0]} at index {i}, expected {dim0}."
                            )
                    summary_tensor = torch.as_tensor(np.stack(summary_rows, axis=0), dtype=torch.float32)
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
        use_subject_summary = bool(getattr(self.params, 'use_subject_summary', False))
        return_metadata = bool(
            getattr(self.params, 'subject_adapter', False)
            or getattr(self.params, 'eeg_channel_context', False)
            or getattr(self.params, 'continual_mode', 'off') != 'off'
            or use_subject_summary
            or (
                getattr(self.params, 'moe', False)
                and getattr(self.params, 'moe_route_mode', '') == 'typed_capacity_domain'
            )
        )
        meta_maps = build_faced_meta_maps(
            self.datasets_dir,
            getattr(self.params, 'faced_meta_csv', ''),
            getattr(self.params, 'subject_summary_file', ''),
            use_subject_summary=use_subject_summary,
        ) if return_metadata else {}
        if return_metadata:
            produced_fields = [
                "subject_id",
                "dataset_id",
                "cohort_id",
                "sample_rate_group_id",
                "age_bucket_id",
                "segment_bucket_id",
                "channel_count",
            ]
            if use_subject_summary:
                produced_fields.append("subject_summary")
            print(
                "[FACED meta] produced_fields="
                f"{produced_fields} use_subject_summary={use_subject_summary}",
                flush=True,
            )

        train_set = CustomDataset(
            self.datasets_dir,
            mode='train',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
            use_subject_summary=use_subject_summary,
        )
        val_set = CustomDataset(
            self.datasets_dir,
            mode='val',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
            use_subject_summary=use_subject_summary,
        )
        test_set = CustomDataset(
            self.datasets_dir,
            mode='test',
            return_keys=rk,
            return_metadata=return_metadata,
            meta_maps=meta_maps,
            use_subject_summary=use_subject_summary,
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
