#!/bin/bash

set -e

START=0
END=1
N_SAMPLES=5000

for i in $(seq $START $END); do
    date=$(date --iso-8601=seconds)
    logfile=logs/$i-$date.log
    echo "Logfile: $logfile"

    RUST_LOG=warn cargo run --release --bin main -- \
        --file data/rise/start_and_goal.csv \
        --iter-distance 3 \
        --rewrite-system rise \
        --n-chains 100000 \
        --chain-length 100 \
        --expr-id $i \
        &>$logfile #

    echo "Finished $i"
done

# --profile=release-with-debug for profiling
# sample-with-baseline \
# --random-guides 5000 \
# --random-goals 5 \
