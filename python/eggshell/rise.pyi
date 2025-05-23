# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import builtins
import eggshell
import typing

class PartialRecExpr:
    r"""
    Wrapper type for Python
    """
    used_tokens: builtins.int
    def __new__(cls,token_list:typing.Sequence[builtins.str]): ...
    def to_data(self) -> TreeData:
        ...

    def __str__(self) -> builtins.str:
        ...

    def __repr__(self) -> builtins.str:
        ...

    def to_dot(self, name:builtins.str, path:builtins.str, transparent:builtins.bool=False) -> None:
        ...

    @staticmethod
    def count_expected_tokens(token_list:typing.Sequence[builtins.str]) -> builtins.int:
        ...

    def lower_meta_level(self) -> RecExpr:
        ...


class RecExpr:
    r"""
    Wrapper type for Python
    """
    def __new__(cls,s_expr_str:builtins.str): ...
    def __str__(self) -> builtins.str:
        ...

    def __repr__(self) -> builtins.str:
        ...

    def to_dot(self, name:builtins.str, path:builtins.str, transparent:builtins.bool=False) -> None:
        ...

    def arity(self, position:builtins.int) -> builtins.int:
        ...

    def to_data(self) -> TreeData:
        ...

    @staticmethod
    def from_data(tree_data:TreeData) -> RecExpr:
        ...


def eqsat_check(start:RecExpr, goal:RecExpr, iter_limit:builtins.int) -> tuple[builtins.int, builtins.str, builtins.str]:
    ...

def many_eqsat_check(starts:typing.Sequence[RecExpr], goal:RecExpr, iter_limit:builtins.int) -> builtins.list[tuple[builtins.int, builtins.str, builtins.str]]:
    ...

def name_to_id(s:builtins.str) -> typing.Optional[builtins.int]:
    ...

def num_symbols() -> builtins.int:
    ...

def operators() -> builtins.list[builtins.str]:
    ...

