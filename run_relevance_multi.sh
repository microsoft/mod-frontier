#!/usr/bin/env bash
# Grade response relevance across several independent Azure gpt-4o resources concurrently.
# Each shard runs the `graders` package against a different endpoint (separate quota),
# then outputs are concatenated back in original row order.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA=data/toxicchat_with_v10_grader.jsonl
WORK=data/relshards
PY="${PYTHON_BIN:-python}"
WORKERS=10

mkdir -p "$WORK"
rm -f "$WORK"/shard_*.jsonl "$WORK"/out_*.jsonl "$WORK"/log_*.txt

# endpoint|deployment, one per shard.
# Replace these with your own Azure OpenAI resources (each has separate quota),
# or set a single endpoint via the GRADERS_AZURE_ENDPOINT env var and use one shard.
ENDPOINTS=(
  "https://<your-resource-1>.openai.azure.com|gpt-4o"
  "https://<your-resource-2>.openai.azure.com|gpt-4o"
  "https://<your-resource-3>.openai.azure.com|gpt-4o"
  "https://<your-resource-4>.openai.azure.com|gpt-4o"
  "https://<your-resource-5>.openai.azure.com|gpt-4o"
)
N=${#ENDPOINTS[@]}

# Split the data into N line-contiguous shards (shard_00.jsonl .. shard_0{N-1}.jsonl)
split -n "l/$N" -d --additional-suffix=.jsonl "$DATA" "$WORK/shard_"
shards=( "$WORK"/shard_*.jsonl )

pids=()
for i in "${!ENDPOINTS[@]}"; do
  ep="${ENDPOINTS[$i]%|*}"; dep="${ENDPOINTS[$i]#*|}"
  shard="${shards[$i]}"
  out="$WORK/out_$(printf '%02d' "$i").jsonl"
  log="$WORK/log_$(printf '%02d' "$i").txt"
  ( cd Graders && GRADERS_AZURE_ENDPOINT="$ep" GRADERS_DEPLOYMENT="$dep" \
      "$PY" -m graders grade -g relevance_v01 -f model_output -pf user_input -w "$WORKERS" \
      -i "$REPO_ROOT/$shard" \
      -o "$REPO_ROOT/$out" ) > "$log" 2>&1 &
  pids+=("$!")
  echo "shard $i ($(wc -l < "$shard") rows) -> $ep [$dep] pid ${pids[$i]}"
done

echo "waiting for $N shards..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "all shards finished (fail=$fail)"

# Concatenate in shard order to restore original row order
cat "$WORK"/out_*.jsonl > data/toxicchat_with_relevance.jsonl
total=$(wc -l < data/toxicchat_with_relevance.jsonl)
src=$(wc -l < "$DATA")
missing=$("$PY" -c "import json,sys; print(sum(1 for l in open('data/toxicchat_with_relevance.jsonl') if l.strip() and json.loads(l).get('relevance_score') is None))")
echo "merged -> data/toxicchat_with_relevance.jsonl  rows=$total/$src  none_scores=$missing"
