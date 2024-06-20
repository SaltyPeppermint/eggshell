# Eggshell

A simple wrapper around the Rust [egg](https://egraphs-good.github.io) library for E-Graphs and Equality Saturation with an implementation of the [Halide](https://halide-lang.org) term solver.

Many thanks to the work done by the [caviar](https://github.com/caviar-trs/caviar/blob/main/src/trs/mod.rs) team and [extraction-gym](https://github.com/egraphs-good/extraction-gym) members.

## Building

To build the Rust library and the Rust commandline binary, just run
```
cargo build
```

To build the Python package use maturin in a local virtual environment.
```
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin build
```

You can immediately install the built package into the local venv with
```
maturin develop
```