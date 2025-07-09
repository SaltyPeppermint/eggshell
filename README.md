# Eggshell

A library around the Rust [egg](https://egraphs-good.github.io) library for E-Graphs and Equality Saturation with an implementation of a number of Rewrite Systems, including [Halide](https://halide-lang.org), [Rise](https://rise-lang.org), and Linear Arithmatic.

It provides typed Python bindings (including functionality to convert the AST of terms into matrix representation for Machine Learning), sampling from E-Graphs, and Meta-Languages that add Sketches, Probabilities, and Partial Terms to the Language of Rewrite Systems and their Expressions.

## Building

To build the Rust library and the Rust cmd binary, just run

```sh
cargo build
```

To build the Python package use maturin in a local virtual environment.

```sh
uv venv --python 3.12
source .venv/bin/activate
cargo build --release
cargo run --bin stub_gen
uvx maturin build --release
```
