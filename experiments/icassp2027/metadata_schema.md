# Common metadata schema

Every ICASSP dataset adapter must produce one row per serialized sample with
the following columns:

| Column | Required | Meaning |
|---|---|---|
| `sample_key` | yes | Stable serialized sample identifier |
| `container_key` | yes | Loader-level key selected by a split manifest; equals `sample_key` except for grouped sequence data |
| `subject_id` | yes | Subject/group identifier used for splitting |
| `session_id` | no | Session identifier when available |
| `recording_id` | no | Recording/trial identifier when available |
| `label` | yes | Integer downstream class label |
| `existing_split` | no | Original dataset split, retained only for audit |

For ISRUC, one loader sample is a sequence file containing multiple labelled
epochs. The metadata extractor may emit one row per epoch using a composite
`sample_key`, but all such rows must share the same `container_key`; manifests
must contain `container_key` values, not synthetic epoch keys.

The metadata extractor must fail rather than invent a subject ID. In
particular, PhysioNet-MI must use a verified key-to-subject mapping or a
preprocessing-generated sidecar; index ranges are not an acceptable substitute.

Required audit outputs per dataset:

```text
all_samples.csv
metadata_audit.json
split_manifest.csv
split_manifest.sha256
subject_counts.json
sample_counts.json
class_counts.json
overlap_audit.json
key_existence_audit.json
split_generation.json
```
