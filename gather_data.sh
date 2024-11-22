#!/bin/bash

set -e

START=0
END=2
uuid=$(uuidgen)

echo "UUID of run: $uuid"

for i in $(seq $START $END); do
    RUST_LOG=info samply record cargo run --release -- --file data/rise/guided_eqsat.csv --eclass-samples 2 --memory-limit 1000000000 --time-limit 120 --uuid $uuid --trs rise --expr-id $i &>logs/$i.log
    echo "Finished $i"
done
