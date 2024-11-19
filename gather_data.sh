#!/bin/bash

set -e

START=0
END=2
uuid=$(uuidgen)

for i in $(seq $START $END); do
    RUST_LOG=info cargo run --release -- --file data/dataset/5k_dataset_new_syntax.json --eclass-samples 1000 --memory-limit 1000000000 --time-limit 120 --uuid $uuid --trs halide --seed-term-id $i &>logs/$i.log
    echo "Finished $i"
done
