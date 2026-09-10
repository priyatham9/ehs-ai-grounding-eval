#!/usr/bin/env bash
# Run the Anthropic API adapter across both models and both arms, then build
# a markdown summary. This is the one command that turns a real Anthropic API
# key into a reproducible, publishable benchmark run.
#
# Refuses to run without ANTHROPIC_API_KEY: this script makes real network
# calls to api.anthropic.com and spends real money, so it must never run by
# accident.
set -euo pipefail

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "run_baselines.sh: ANTHROPIC_API_KEY is not set. Refusing to run: this" >&2
    echo "script makes real, billed calls to the Anthropic API. Set the key and" >&2
    echo "re-run, e.g.:" >&2
    echo "  export ANTHROPIC_API_KEY=sk-..." >&2
    echo "  scripts/run_baselines.sh" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DATE="$(date -u +%Y%m%d)"
RUN_ID="baseline_run_${RUN_DATE}"
OUT_DIR="results/${RUN_ID}"
PUBLISHED_DIR="results/published"

mkdir -p "$OUT_DIR" "$PUBLISHED_DIR"

# Model ids as of the claude-api skill's current table (checked before writing
# this script): Claude Haiku 4.5 is `claude-haiku-4-5` (no date suffix in the
# current API - current model ids are not date-suffixed) and Claude Sonnet 5
# is `claude-sonnet-5`. Adjust here if the skill's table changes.
MODELS=("claude-haiku-4-5" "claude-sonnet-5")
ARMS=("ungrounded" "grounded")

echo "run_baselines.sh: run id ${RUN_ID}, writing into ${OUT_DIR}/"
echo

for MODEL in "${MODELS[@]}"; do
    for ARM in "${ARMS[@]}"; do
        GROUNDED_FLAG=()
        if [ "$ARM" = "grounded" ]; then
            GROUNDED_FLAG=(--grounded)
        fi

        echo "=== model=${MODEL} arm=${ARM} ==="
        echo "--- dry-run token estimate ---"
        python3 -m grounding_eval.cli run \
            --adapter anthropic --model "$MODEL" "${GROUNDED_FLAG[@]}" --dry-run

        echo "--- real run ---"
        python3 -m grounding_eval.cli run \
            --adapter anthropic --model "$MODEL" "${GROUNDED_FLAG[@]}" \
            --out "$OUT_DIR"
        echo
    done
done

echo "=== summary ==="
python3 -m grounding_eval.cli report "$OUT_DIR" | tee "${OUT_DIR}/summary.md"

echo
echo "=== publishing finished runs to ${PUBLISHED_DIR}/ ==="
for f in "$OUT_DIR"/*.json; do
    [ -e "$f" ] || continue
    cp -v "$f" "$PUBLISHED_DIR"/
done

echo
echo "Done. Run files are in ${OUT_DIR}/ and ${PUBLISHED_DIR}/."
echo "Review ${PUBLISHED_DIR}/ before committing: see results/published/README.md"
echo "for what a published run file must carry."
