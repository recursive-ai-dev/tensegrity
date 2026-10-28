"""Typed AST for the symbolic language."""
from dataclasses import dataclass
from typing import Dict, Tuple, Union


@dataclass(frozen=True)
class IntT:
    pass


@dataclass(frozen=True)
class BoolT:
    pass


@dataclass(frozen=True)
class ListT:
    elem: "Type"


@dataclass(frozen=True)
class FunT:
    arg: "Type"
    ret: "Type"


Type = Union[IntT, BoolT, ListT, FunT]


@dataclass(frozen=True)
class Lit:
    value: Union[int, bool]


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Prim:
    name: str


@dataclass(frozen=True)
class App:
    fn: "Expr"
    args: Tuple["Expr", ...]


@dataclass(frozen=True)
class Lambda:
    params: Tuple[Tuple[str, Type], ...]
    body: "Expr"


@dataclass(frozen=True)
class Let:
    bindings: Tuple[Tuple[str, "Expr"], ...]
    body: "Expr"


@dataclass(frozen=True)
class If:
    cond: "Expr"
    then: "Expr"
    other: "Expr"


@dataclass(frozen=True)
class Compose:
    f: "Expr"
    g: "Expr"


@dataclass(frozen=True)
class MapE:
    f: "Expr"
    xs: "Expr"


@dataclass(frozen=True)
class FilterE:
    f: "Expr"
    xs: "Expr"


@dataclass(frozen=True)
class FoldE:
    f: "Expr"
    init: "Expr"
    xs: "Expr"


@dataclass(frozen=True)
class ListLit:
    items: Tuple["Expr", ...]


@dataclass(frozen=True)
class Nil:
    elem: Type


@dataclass(frozen=True)
class Hole:
    pass


@dataclass(frozen=True)
class Agent:
    params: Tuple[Tuple[str, Type], ...]
    out: Type
    body: "Expr"


Expr = Union[Lit, Var, Prim, App, Lambda, Let, If, Compose, MapE, FilterE,
             FoldE, ListLit, Nil, Hole, Agent]

PRIMS: Dict[str, Tuple[int, ...]] = {
    "add": (2,), "sub": (2,), "mul": (2,), "eq": (2,), "lt": (2,),
    "le": (2,), "and": (2,), "or": (2,), "cons": (2,),
    "neg": (1,), "not": (1,), "head": (1,), "tail": (1,), "len": (1,),
}


def children(e: Expr) -> Tuple[Expr, ...]:
    """Direct sub-expressions in left-to-right order."""
    if isinstance(e, App):
        return (e.fn,) + tuple(e.args)
    if isinstance(e, (Lambda, Agent)):
        return (e.body,)
    if isinstance(e, Let):
        return tuple(b for _, b in e.bindings) + (e.body,)
    if isinstance(e, If):
        return (e.cond, e.then, e.other)
    if isinstance(e, Compose):
        return (e.f, e.g)
    if isinstance(e, (MapE, FilterE)):
        return (e.f, e.xs)
    if isinstance(e, FoldE):
        return (e.f, e.init, e.xs)
    if isinstance(e, ListLit):
        return tuple(e.items)
    return ()


def size(e: Expr) -> int:
    return 1 + sum(size(c) for c in children(e))


def depth(e: Expr) -> int:
    return 1 + max((depth(c) for c in children(e)), default=0)
