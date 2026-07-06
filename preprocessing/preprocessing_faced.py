import scipy
from scipy import signal
import os
import lmdb
import pickle
import json
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from utils.faced_channel_manifest import build_faced_channel_manifest

labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])
root_dir = '/data/cyn/FACED/Processed_data'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train':files[:80],
    'val':files[80:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

output_root = '/data/datasets/BigDownstream/Faced/processed'
db = lmdb.open(output_root, map_size=6612500172)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        f = open(os.path.join(root_dir, file), 'rb')
        array = pickle.load(f)
        eeg = signal.resample(array, 6000, axis=2)
        eeg_ = eeg.reshape(28, 32, 30, 200)
        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10*j:10*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
manifest = build_faced_channel_manifest()
txn.put(key=b'__channel_manifest__', value=json.dumps(manifest, ensure_ascii=True).encode('utf-8'))
txn.put(key=b'__channel_names__', value=pickle.dumps(manifest['labram_channel_names']))
txn.put(key=b'channel_names', value=pickle.dumps(manifest['labram_channel_names']))
txn.put(key=b'ch_names', value=pickle.dumps(manifest['labram_channel_names']))
txn.commit()
db.close()

with open(os.path.join(output_root, 'channel_manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=True)
