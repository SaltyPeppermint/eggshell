#!/bin/bash

set -e

START=0
END=1
N_SAMPLES=20000

for i in $(seq $START $END); do
    date=$(date --iso-8601=seconds)
    logfile=logs/$i-$date.log
    echo "Logfile: $logfile"

    RUST_LOG=info cargo run --release -- \
        --file data/rise/start_and_goal.csv \
        --eclass-samples $N_SAMPLES \
        --memory-limit 1000000000 \
        --rewrite-system rise \
        --expr-id $i \
        &>$logfile #

    echo "Finished $i"
done

# --profile=release-with-debug for profiling
# sample-with-baseline \
# --random-guides 5000 \
# --random-goals 5 \
