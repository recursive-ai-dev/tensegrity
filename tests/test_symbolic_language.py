from pathlib import Path

from tensegrity.symbolic.algebra import App, Lit, Prim, depth, size, children
from tensegrity.symbolic.errors import ParseError, TypeError_, EvalError


def test_errors_carry_kind_and_code():
    assert ParseError("x").kind == "parse"
    assert TypeError_("y", "d").kind == "type"
    e = EvalError("z")
    assert e.kind == "eval" and e.code == "z"


def test_nodes_are_hashable_and_equal_by_value():
    a = App(Prim("add"), (Lit(1),))
    b = App(Prim("add"), (Lit(1),))
    assert a == b
    assert len({a, b}) == 1


def test_children_size_depth():
    e = App(Prim("add"), (Lit(1), Lit(2)))
    assert children(e) == (Prim("add"), Lit(1), Lit(2))
    assert size(e) == 4
    assert depth(Lit(1)) == 1
    assert depth(e) == 2


def test_pyproject_lists_symbolic_package():
    root = Path(__file__).resolve().parent.parent
    assert "tensegrity.symbolic" in (root / "pyproject.toml").read_text()
