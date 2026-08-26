# Archived ICASSP development paths

The former DepthAgg and Static/Routed ICASSP development paths are archived. Their launchers and contract tests have been removed from the active tree because they no longer define the paper we intend to submit.

The corresponding source history, logs, and output directories are retained for reproducibility and artifact-audit purposes. They must not be launched, aggregated, or described as the primary evidence for the focused revision.

In particular:

- `output/icassp2027_depth` is historical pilot evidence, not the main result table;
- subject-disjoint 20-epoch pilots must not be mixed with the rejected-paper benchmark protocol;
- old routing outputs may be promoted only after their split, seed, commit, checkpoint, selection rule, and TMLR-overlap status are verified;
- recovery of a deleted file should be done from Git history, not by recreating an obsolete active contract.

The active experiment definition is [REVISION_CONTRACT.md](REVISION_CONTRACT.md).
