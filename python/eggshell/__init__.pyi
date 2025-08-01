# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import builtins
import typing
from . import arithmetic
from . import halide
from . import rise
from . import simple

class FirstErrorDistance:
    hits: builtins.list[builtins.int]
    hit_probabilities: builtins.list[typing.Optional[builtins.float]]
    misses: builtins.list[builtins.int]
    miss_probabilities: builtins.list[typing.Optional[builtins.float]]
    def n_hits(self) -> builtins.int: ...
    def n_misses(self) -> builtins.int: ...
    def combine(self, rhs:FirstErrorDistance) -> None: ...
    def extend(self, others:typing.Sequence[FirstErrorDistance]) -> None: ...

class Node:
    raw_name: builtins.str
    arity: builtins.int
    nth_child: builtins.int
    bfs_order: builtins.int
    depth: builtins.int
    id: builtins.int
    value: typing.Optional[builtins.str]
    name: builtins.str
    def arity(self) -> builtins.int: ...
    def depth(self) -> builtins.int: ...

class TreeData:
    def transposed_adjacency(self) -> builtins.list[builtins.list[builtins.int]]: ...
    def anc_matrix(self, max_rel_distance:builtins.int) -> builtins.list[builtins.list[builtins.int]]:
        r"""
        Gives a matrix that describes the relationship of an ancestor to a child as a distance between them
        maximum distance (positive or negative) to be encoded mapped to the range 2 * max_rel_distance
        If the distance is too large or no relationship exists, 0 is returned
        """
    def sib_matrix(self, max_rel_distance:builtins.int) -> builtins.list[builtins.list[builtins.int]]:
        r"""
        Gives a matrix that describes the sibling relationship in nodes
        max_relative_distance describes the maximum distance (positive or negative) to be encoded,
        mapped to the range 2 * max_relative_distance
        If the distance is too large or no relationship exists, 0 is returned
        """
    def count_symbols(self, n_symbols:builtins.int, n_vars:builtins.int) -> builtins.list[builtins.int]: ...
    def nodes(self) -> builtins.list[Node]: ...
    def values(self) -> builtins.list[builtins.str]: ...
    def names(self) -> builtins.list[builtins.str]: ...
    def arity(self, position:builtins.int) -> builtins.int: ...
    def depth(self) -> builtins.int: ...
    def __len__(self) -> builtins.int: ...
    def simple_feature_names(self, symbol_names:typing.Sequence[builtins.str], var_names:typing.Sequence[builtins.str]) -> builtins.list[builtins.str]: ...
    def simple_features(self, n_symbols:builtins.int, n_vars:builtins.int) -> builtins.list[builtins.float]: ...
    @staticmethod
    def batch_simple_features(tree_datas:typing.Sequence[TreeData], n_symbols:builtins.int, n_vars:builtins.int) -> builtins.list[builtins.list[builtins.float]]: ...

