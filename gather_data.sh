#!/bin/bash

set -e

START=0
END=0
N_SAMPLES=20000
uuid=$(uuidgen)

echo "UUID of run: $uuid"

for i in $(seq $START $END); do
    logfile=logs/$i-$(date +%H:%M:%S).log
    echo "Logfile: $logfile"

    RUST_LOG=info cargo run --profile=release-with-debug -- \
        --file data/rise/guided_eqsat.csv \
        --eclass-samples $N_SAMPLES \
        --memory-limit 1000000000 \
        --random-guides 5000 \
        --random-goals 5 \
        --time-limit 120 \
        --uuid $uuid \
        --trs rise \
        --expr-id $i &>$logfile # --random-guide-generation 3 \

    echo "Finished $i"
done

# --profile=release-with-debug for profiling
