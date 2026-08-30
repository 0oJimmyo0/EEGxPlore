# ICASSP confirmatory analysis

These scripts are analysis-only. The confirmatory training rows remain pinned
to the frozen training commit `fd425cdfd0ff08d57ac30ee9b8737b895e9d46ad`.

After the 24 paper-facing cells finish, run:

```bash
python experiments/icassp2027/analysis/audit_confirmatory_matrix.py --strict
python experiments/icassp2027/analysis/aggregate_confirmatory_results.py
python experiments/icassp2027/analysis/make_paper_tables.py
python experiments/icassp2027/analysis/aggregate_routing_validation.py
```

The strict audit reads only the frozen FACED/ISRUC/SEED-V/PhysioNet-MI matrix from
`paper_table_manifest_v3.csv`. It rejects incomplete rows, seed 42, smoke
runs, historical candidates, and non-confirmatory conditions. The routing
summary is explicitly scoped to the stored last validation-batch snapshot for
each epoch; it is not a complete validation-set aggregation and must not be
used to claim globally balanced routing.
