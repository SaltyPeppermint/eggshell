# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import typing

class PyGraphData:
    nodes: list[int]
    const_values: list[typing.Optional[float]]
    edges: list[list[int]]
    ignore_unknown: bool
    def __new__(cls,rec_expr:PyRecExpr, variable_names:typing.Sequence[str], ignore_unknown:bool): ...
    @staticmethod
    def batch_new(rec_exprs:typing.Sequence[PyRecExpr], variable_names:typing.Sequence[str], ignore_unknown:bool) -> list[PyGraphData]:
        ...

    def to_rec_expr(self, variable_names:typing.Sequence[str]) -> PyRecExpr:
        ...

    @staticmethod
    def num_node_types(variable_names:typing.Sequence[str], ignore_unknown:bool) -> int:
        ...


class PyRecExpr:
    r"""
    Wrapper type for Python
    """
    def __new__(cls,s_expr_str:str): ...
    @staticmethod
    def batch_new(s_expr_strs:typing.Sequence[str]) -> list[PyRecExpr]:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def arity(self, position:int) -> int:
        ...

    def size(self) -> int:
        ...

    def depth(self) -> int:
        ...

    def count_symbols(self, variable_names:typing.Sequence[str], ignore_unknown:bool) -> list[int]:
        ...

    def simple_features(self, variable_names:typing.Sequence[str], ignore_unknown:bool) -> list[float]:
        ...

    @staticmethod
    def batch_simple_features(exprs:typing.Sequence[PyRecExpr], variable_names:typing.Sequence[str], ignore_unknown:bool) -> list[list[float]]:
        ...

    @staticmethod
    def simple_feature_names(var_names:typing.Sequence[str], ignore_unknown:bool) -> list[str]:
        ...


def eqsat_check(start:PyRecExpr,goal:PyRecExpr,iter_limit:int) -> tuple[int, str, str]:
    ...

def many_eqsat_check(starts:typing.Sequence[PyRecExpr],goal:PyRecExpr,iter_limit:int) -> list[tuple[int, str, str]]:
    ...

