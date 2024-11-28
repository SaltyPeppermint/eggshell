#!/bin/bash

set -e

START=0
END=0
N_SAMPLES=200000
uuid=$(uuidgen)

echo "UUID of run: $uuid"

for i in $(seq $START $END); do
    RUST_LOG=info cargo run --profile=release-with-debug -- --file data/rise/guided_eqsat.csv --eclass-samples $N_SAMPLES --memory-limit 1000000000 --time-limit 120 --uuid $uuid --trs rise --expr-id $i &>logs/$i.log
    echo "Finished $i"
done

# --profile=release-with-debug for profiling
