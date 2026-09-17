#!/usr/bin/env bash
# cluster/run.sh — Full pipeline: sync code → submit job → poll → sync results
set -euo pipefail
source "$(dirname "$0")/config.sh"

SCRIPT="${1:?Usage: $0 <script_id> [--no-poll]}"
NO_POLL=false
if [[ "${2:-}" == "--no-poll" ]]; then
    NO_POLL=true
fi

JOB_SCRIPT="cluster/jobs/${SCRIPT}.sh"

if [ ! -f "${LOCAL_PROJECT}/${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found: ${LOCAL_PROJECT}/${JOB_SCRIPT}"
    echo "Available job scripts:"
    ls "${LOCAL_PROJECT}/cluster/jobs/"
    exit 1
fi

# ── Step 1: Sync code ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Step 1: Syncing code to cluster"
echo "=========================================="
bash "$(dirname "$0")/sync-code.sh"

# ── Step 2: Submit job ───────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Step 2: Submitting ${SCRIPT} to SLURM"
echo "=========================================="

JOB_ID=$(ssh "$CLUSTER_HOST" bash -s <<SUBMIT
set -euo pipefail
cd "${CLUSTER_PROJECT}"
mkdir -p cluster/logs

# Submit and extract job ID
OUTPUT=\$(sbatch "${JOB_SCRIPT}" 2>&1)
echo "\$OUTPUT" >&2
# Extract numeric job ID from "Submitted batch job 12345"
echo "\$OUTPUT" | grep -oP '\\d+' | tail -1
SUBMIT
)

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job — no job ID returned"
    exit 1
fi

echo "Submitted job: ${JOB_ID}"

if $NO_POLL; then
    echo ""
    echo "Fire-and-forget mode. Check status with:"
    echo "  make cluster-status"
    echo "  make cluster-sync-down SCRIPT=${SCRIPT%%[a-z]*}"
    exit 0
fi

# ── Step 3: Poll until complete ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Step 3: Polling job ${JOB_ID}"
echo "=========================================="

POLL_INTERVAL=$POLL_INITIAL

while true; do
    # Check if any tasks are still running/pending
    REMAINING=$(ssh "$CLUSTER_HOST" "squeue -j ${JOB_ID} -h -t RUNNING,PENDING 2>/dev/null | wc -l" || echo "0")
    REMAINING=$(echo "$REMAINING" | tr -d '[:space:]')

    if [ "$REMAINING" -eq 0 ] 2>/dev/null; then
        echo ""
        echo "All tasks completed."
        break
    fi

    echo "  [$(date +%H:%M:%S)] ${REMAINING} task(s) still running — next check in ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"

    # Escalate interval
    POLL_INTERVAL=$(python3 -c "print(min(int(${POLL_INTERVAL} * ${POLL_BACKOFF}), ${POLL_MAX}))")
done

# ── Step 4: Check for failures ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Step 4: Checking job outcomes"
echo "=========================================="

ssh "$CLUSTER_HOST" bash -s <<CHECK
set -euo pipefail
echo "sacct for job ${JOB_ID}:"
sacct -j "${JOB_ID}" --format="JobID%15,JobName%15,State%12,ExitCode,Elapsed,MaxRSS" 2>/dev/null || echo "(sacct unavailable)"

FAILED=\$(sacct -j "${JOB_ID}" --format="State" --noheader 2>/dev/null | grep -c "FAILED" || true)
if [ "\$FAILED" -gt 0 ]; then
    echo ""
    echo "WARNING: \${FAILED} task(s) FAILED. Check logs:"
    echo "  make cluster-logs SCRIPT=${SCRIPT}"
fi
CHECK

# ── Step 5: Sync results ────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Step 5: Syncing results"
echo "=========================================="
# Extract analysis prefix (e.g., "11" from "11a")
ANALYSIS_PREFIX="${SCRIPT%%[a-z]*}"
bash "$(dirname "$0")/sync-results.sh" "$ANALYSIS_PREFIX"

echo ""
echo "=========================================="
echo "  Pipeline complete for ${SCRIPT}"
echo "=========================================="
