# ICASSP confirmatory analysis

These scripts are analysis-only. The confirmatory training rows remain pinned
to the frozen training commit `fd425cdfd0ff08d57ac30ee9b8737b895e9d46ad`.

After the 24 paper-facing rows are frozen, run:

```bash
python experiments/icassp2027/analysis/audit_confirmatory_matrix.py --strict
python experiments/icassp2027/analysis/aggregate_confirmatory_results.py
python experiments/icassp2027/analysis/make_paper_tables.py
python experiments/icassp2027/analysis/aggregate_routing_validation.py
```

For the external-workspace snapshot used by the final paper, the strict audit
can be run directly against the frozen root:

```bash
python experiments/icassp2027/analysis/audit_confirmatory_matrix.py \
  --output-root output/icassp2027_frozen_20260830 --strict
```

The complete validation routing pass is separate from the historical
last-batch summary and uses no training:

```bash
sbatch experiments/icassp2027/analysis/submit_full_validation_routing.slurm
```

The strict audit reads only the frozen FACED/ISRUC/SEED-V/PhysioNet-MI matrix from
`paper_table_manifest_v4.csv`. It rejects incomplete rows, seed 42, smoke
runs, historical candidates, and non-confirmatory conditions. The legacy
routing summary is explicitly scoped to the stored last validation-batch
snapshot for each epoch; it is not a complete validation-set aggregation and
must not be used to claim globally balanced routing. The new
`full_validation_routing_diagnostics.py` output is the complete-validation
artifact for that purpose, but remains supplementary diagnostic evidence
rather than a paper metric.
