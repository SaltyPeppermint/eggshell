# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import numpy
import numpy.typing
import typing

class PyLanguageManager:
    r"""
    Wrapper type for Python
    """
    def __new__(cls,variable_names,ignore_unknown = ...): ...
    def feature_names_simple(self) -> list[str]:
        ...

    def featurize_simple(self, expr:PyRecExpr, lang_manager:PyLanguageManager) -> numpy.typing.NDArray[numpy.float64]:
        ...

    def many_featurize_simple(self, exprs:typing.Sequence[PyRecExpr], lang_manager:PyLanguageManager) -> numpy.typing.NDArray[numpy.float64]:
        ...


class PyNode:
    def __new__(cls,node_name:str, children:typing.Sequence[int]): ...
    def name(self) -> str:
        ...

    def is_leaf(self) -> bool:
        ...

    def children(self) -> list[int]:
        ...

    def leaf_feature(self, lang_manager:PyLanguageManager) -> list[float]:
        ...

    def non_leaf_id(self, lang_manager:PyLanguageManager) -> int:
        ...


class PyRecExpr:
    r"""
    Wrapper type for Python
    """
    def __new__(cls,s_expr_str:str): ...
    @staticmethod
    def many_new(s_expr_strs:typing.Sequence[str]) -> list[PyRecExpr]:
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

    def count_symbols(self, lang_manager:PyLanguageManager) -> list[int]:
        ...

    def feature_vec_simple(self, lang_manager:PyLanguageManager) -> list[float]:
        ...


def eqsat_check(start:PyRecExpr,goal:PyRecExpr) -> tuple[int, str, str]:
    ...

def many_eqsat_check(starts:typing.Sequence[PyRecExpr],goal:PyRecExpr) -> list[tuple[int, str, str]]:
    ...
