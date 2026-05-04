#!/bin/bash
# OmniCoder-2-9B Benchmark — moxing serve only
# Usage: bash scripts/bench_omnicoder.sh
set -e

MODELS=(
    "omnicoder-2-9b-q4_k_m.gguf"
    "omnicoder-2-9b-q5_k_m.gguf"
)
BACKENDS=(cuda vulkan cpu)
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "# OmniCoder-2-9B Benchmark"
echo "**MoXing:** $(python -c 'from moxing import __version__; print(__version__)')"
echo "**GPU:** $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo ""
echo "| Model | Backend | tok/s |"
echo "|-------|---------|-------|"

for MODEL in "${MODELS[@]}"; do
    for BACKEND in "${BACKENDS[@]}"; do
        PORT=20050
        while lsof -i :$PORT >/dev/null 2>&1; do PORT=$((PORT+1)); done

        moxing serve "$DIR/$MODEL" -b "$BACKEND" -p "$PORT" -c 1024 >/dev/null 2>&1 &
        PID=$!

        for i in $(seq 1 30); do
            sleep 2
            curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            kill -0 $PID 2>/dev/null || break
        done

        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
            echo "| $(basename $MODEL .gguf) | $BACKEND | FAIL |"
            kill $PID 2>/dev/null; wait $PID 2>/dev/null; continue
        }

        T0=$(date +%s.%N)
        curl -s --max-time 60 "http://127.0.0.1:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Write fibonacci memoization in Python.","n_predict":128,"temperature":0}' >/dev/null
        T1=$(date +%s.%N)

        TOK=$(curl -s --max-time 60 "http://127.0.0.1:$PORT/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"ok","n_predict":64,"temperature":0}' | \
            python3 -c "import json,sys;print(json.load(sys.stdin).get('tokens_predicted',0))" 2>/dev/null || echo 0)

        ELAPSED=$(python3 -c "print(round($T1-$T0,2))")
        TPS=$(python3 -c "print(round($TOK/$ELAPSED,1))" 2>/dev/null || echo "-")

        kill $PID 2>/dev/null; wait $PID 2>/dev/null
        NAME=$(echo $MODEL | sed 's/-q/_q/' | sed 's/.gguf//')
        echo "| $NAME | **$BACKEND** | **$TPS** |"
        sleep 2
    done
done

echo ""
echo "**Best:** \`moxing serve omnicoder-2-9b-q4_k_m.gguf -b cuda\` (~41 tok/s)"
echo "**Quality:** \`moxing serve omnicoder-2-9b-q5_k_m.gguf -b cuda\` (~37 tok/s, richer reasoning)"
echo "**vLLM:** NOT supported (qwen35 architecture)"
