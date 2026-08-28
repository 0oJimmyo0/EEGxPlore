#!/usr/bin/env bash
set -euo pipefail

# Development-only wrapper for the historical FACED/ISRUC candidate recipes.
# It intentionally routes through run_revision.sh so data-contract, clean-tree,
# and manifest safeguards remain shared with the paper launcher.
#
# Usage:
#   run_historical_candidate.sh DATASET {opt|route} SEED [smoke|full] [MODEL_ROOT] [EXPECTED_COMMIT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -lt 3 || "$#" -gt 6 ]]; then
  echo "Usage: $0 DATASET {opt|route} SEED [smoke|full] [MODEL_ROOT] [EXPECTED_COMMIT]" >&2
  exit 2
fi

DATASET="$1"
STAGE="$2"
SEED="$3"
EXECUTION_MODE="${4:-full}"
MODEL_ROOT_OVERRIDE="${5:-${MODEL_ROOT:-}}"
EXPECTED_COMMIT="${6:-${EXPECTED_COMMIT:-$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD)}}"

case "$DATASET:$STAGE" in
  FACED:opt) RECIPE_PATH="$SCRIPT_DIR/historical_candidate_faced_opt_v1.json" ;;
  FACED:route) RECIPE_PATH="$SCRIPT_DIR/historical_candidate_faced_route_v1.json" ;;
  ISRUC:opt) RECIPE_PATH="$SCRIPT_DIR/historical_candidate_isruc_opt_v1.json" ;;
  ISRUC:route) RECIPE_PATH="$SCRIPT_DIR/historical_candidate_isruc_route_v1.json" ;;
  *)
    echo "Unsupported historical candidate: $DATASET/$STAGE" >&2
    exit 2
    ;;
esac

case "$EXECUTION_MODE" in
  smoke)
    HISTORICAL_CANDIDATE_SMOKE=1
    EPOCH_OVERRIDE=1
    ;;
  full)
    HISTORICAL_CANDIDATE_SMOKE=0
    EPOCH_OVERRIDE=""
    ;;
  *)
    echo "execution mode must be smoke or full (got: $EXECUTION_MODE)" >&2
    exit 2
    ;;
esac

CORE_ARGS=("$DATASET" historical_candidate "$SEED" cbramod_benchmark)
if [[ -n "$EPOCH_OVERRIDE" ]]; then
  CORE_ARGS+=("$EPOCH_OVERRIDE")
else
  CORE_ARGS+=("")
fi
if [[ -n "$MODEL_ROOT_OVERRIDE" ]]; then
  CORE_ARGS+=("$MODEL_ROOT_OVERRIDE")
else
  CORE_ARGS+=("$SCRIPT_DIR/../../../output/icassp2027_historical_candidate")
fi
CORE_ARGS+=("$EXPECTED_COMMIT")

RUN_MODE=internal \
HISTORICAL_CANDIDATE_STAGE="$STAGE" \
HISTORICAL_CANDIDATE_RECIPE_PATH="$RECIPE_PATH" \
HISTORICAL_CANDIDATE_SMOKE="$HISTORICAL_CANDIDATE_SMOKE" \
"$SCRIPT_DIR/run_revision.sh" "${CORE_ARGS[@]}"
