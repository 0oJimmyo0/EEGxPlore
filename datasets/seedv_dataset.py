from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import lmdb
import pickle
import os
import json


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            return_keys: bool = False,
            split_manifest_path: str = '',
    ):
        super(CustomDataset, self).__init__()
        self.mode = mode
        self.return_keys = return_keys
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        split_index = None
        if split_manifest_path:
            if not os.path.isfile(split_manifest_path):
                raise FileNotFoundError(f"SEED-V split manifest not found: {split_manifest_path}")
            if split_manifest_path.endswith('.json'):
                with open(split_manifest_path, 'r', encoding='utf-8') as f:
                    split_index = json.load(f)
            else:
                with open(split_manifest_path, 'rb') as f:
                    split_index = pickle.load(f)
        else:
            with self.db.begin(write=False) as txn:
                split_index = pickle.loads(txn.get('__keys__'.encode()))

        if mode not in split_index:
            raise KeyError(f"SEED-V LMDB missing split {mode!r}; available splits: {list(split_index.keys())}")
        self.keys = split_index[mode]

        if len(self.keys) == 0:
            raise ValueError(f"SEED-V split {mode!r} is empty in {data_dir}")

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        enc_key = key.encode() if isinstance(key, str) else key
        with self.db.begin(write=False) as txn:
            raw = txn.get(enc_key)
        if raw is None:
            raise KeyError(f"SEED-V LMDB key not found: {key!r}")
        pair = pickle.loads(raw)
        data = pair['sample']
        label = pair['label']
        if self.return_keys:
            kstr = key.decode() if isinstance(key, bytes) else str(key)
            return data / 100, label, kstr
        return data / 100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        if self.return_keys:
            keys = [x[2] for x in batch]
            return to_tensor(x_data), to_tensor(y_label).long(), keys
        return to_tensor(x_data), to_tensor(y_label).long()


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    @staticmethod
    def _parse_key_fields(key):
        k = key.decode() if isinstance(key, bytes) else str(key)
        parts = k.rsplit('-', 2)
        trial = parts[1] if len(parts) == 3 else ''
        prefix = parts[0] if len(parts) == 3 else k
        seg = parts[2] if len(parts) == 3 else ''

        fparts = prefix.split('_')
        subject = fparts[0] if len(fparts) >= 1 else ''
        session = fparts[1] if len(fparts) >= 2 else ''
        return {
            'subject': subject,
            'session': session,
            'trial': trial,
            'segment': seg,
            'key': k,
        }

    def _report_split_protocol(self, train_set, val_set, test_set):
        split_sets = {'train': train_set, 'val': val_set, 'test': test_set}
        subject_sets = {}
        split_trials = {}
        for split_name, ds in split_sets.items():
            subjects = set()
            sessions = set()
            trials = set()
            for key in ds.keys:
                p = self._parse_key_fields(key)
                if p['subject']:
                    subjects.add(p['subject'])
                if p['session']:
                    sessions.add(p['session'])
                if p['trial']:
                    trials.add(p['trial'])
            subject_sets[split_name] = subjects
            split_trials[split_name] = sorted(trials)
            print(
                f"[SEED-V split] {split_name}: n={len(ds)}, subjects={len(subjects)}, "
                f"sessions={sorted(sessions)}, trials={sorted(trials)}"
            )

        tv = len(subject_sets['train'] & subject_sets['val'])
        tt = len(subject_sets['train'] & subject_sets['test'])
        vt = len(subject_sets['val'] & subject_sets['test'])
        print(f"[SEED-V split] subject overlap train/val={tv}, train/test={tt}, val/test={vt}")

        trial_ok = (
            split_trials['train'] == ['0', '1', '2', '3', '4']
            and split_trials['val'] == ['5', '6', '7', '8', '9']
            and split_trials['test'] == ['10', '11', '12', '13', '14']
        )
        if trial_ok:
            print('[SEED-V split] protocol looks trial-stratified per recording (not subject-disjoint).')
        else:
            print('[SEED-V split] non-default trial partition detected; verify against your protocol.')

    def _report_schema(self, dataset):
        one = dataset[0]
        x = one[0]
        y = one[1]
        x_shape = tuple(x.shape) if hasattr(x, 'shape') else 'unknown'
        x_dtype = getattr(x, 'dtype', type(x))
        print(f"[SEED-V schema] sample_shape={x_shape} sample_dtype={x_dtype} label={int(y)}")
        if x_shape != (62, 1, 200):
            print('[SEED-V schema] warning: expected (62, 1, 200); check preprocessing.')

    def get_data_loader(self):
        rk = bool(getattr(self.params, 'return_sample_keys', False))
        split_manifest_path = str(getattr(self.params, 'seedv_split_manifest', '') or '')
        if split_manifest_path:
            print(f"[SEED-V split] using external manifest: {split_manifest_path}")
        train_set = CustomDataset(
            self.datasets_dir,
            mode='train',
            return_keys=rk,
            split_manifest_path=split_manifest_path,
        )
        val_set = CustomDataset(
            self.datasets_dir,
            mode='val',
            return_keys=rk,
            split_manifest_path=split_manifest_path,
        )
        test_set = CustomDataset(
            self.datasets_dir,
            mode='test',
            return_keys=rk,
            split_manifest_path=split_manifest_path,
        )

        self._report_split_protocol(train_set, val_set, test_set)
        self._report_schema(train_set)

        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        num_workers = int(getattr(self.params, 'num_workers', 0))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                num_workers=num_workers,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=num_workers,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=num_workers,
            ),
        }
        return data_loader
