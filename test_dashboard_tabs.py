import ast
from pathlib import Path


def _rendered_tab_indices(source):
    tree = ast.parse(source)
    indices = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue

        for item in node.items:
            expr = item.context_expr
            if (
                isinstance(expr, ast.Subscript)
                and isinstance(expr.value, ast.Name)
                and expr.value.id == "tabs"
                and isinstance(expr.slice, ast.Constant)
                and isinstance(expr.slice.value, int)
            ):
                indices.add(expr.slice.value)

    return indices


def test_dashboard_renders_every_declared_tab():
    source = Path(__file__).with_name("view_results.py").read_text(encoding="utf-8")

    assert "omitted here for brevity" not in source
    assert _rendered_tab_indices(source) == set(range(21))
