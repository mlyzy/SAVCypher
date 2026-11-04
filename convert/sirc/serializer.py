from __future__ import annotations

from sirc.ast import (
    Aggregate,
    AnyClause,
    Dedup,
    Expand,
    Filter,
    Graph,
    Keep,
    Let,
    Node,
    Offset,
    OptionalBlock,
    Order,
    Pattern,
    Query,
    Relationship,
    Scope,
    Take,
    Top,
)
from sirc.parser import parse_sirc


def _expression(text: str) -> str:
    return " ".join(str(text).strip().split())


def _item(item) -> str:
    value = _expression(item.expression.text)
    return f"{value} AS {item.alias}" if item.alias else value


def _properties(properties) -> str:
    if not properties:
        return ""
    return " {" + ", ".join(
        f"{name} = {_expression(value.text)}" for name, value in properties
    ) + "}"


def _node(node: Node) -> str:
    labels = ""
    if node.labels:
        labels = ":" + "&".join(node.labels)
    if node.variable:
        return node.variable + labels + _properties(node.properties)
    return "(" + labels + _properties(node.properties) + ")"


def _relationship(relation: Relationship) -> str:
    value = relation.variable or ""
    if relation.types:
        value += ":" + "|".join(relation.types)
    value += relation.hop or ""
    value += _properties(relation.properties)
    if relation.direction == "in":
        return f"<-{value}-"
    if relation.direction == "out":
        return f"-{value}->"
    return f"-{value}-"


def _pattern(pattern: Pattern) -> str:
    value = _node(pattern.nodes[0])
    for relation, node in zip(pattern.relationships, pattern.nodes[1:]):
        value += " " + _relationship(relation) + " " + _node(node)
    if pattern.path_variable:
        mode = f"{pattern.path_mode} " if pattern.path_mode else ""
        value = f"PATH {pattern.path_variable} = {mode}{value}"
    return value


def _order_items(items) -> str:
    return ", ".join(
        f"{_expression(item.expression.text)} {item.direction.upper()}"
        for item in items
    )


def _indent(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line for line in lines]


def _steps(steps) -> list[str]:
    lines: list[str] = []
    for step in steps:
        if isinstance(step, Graph):
            lines.append("GRAPH " + ";\n  ".join(_pattern(item) for item in step.patterns))
        elif isinstance(step, Filter):
            lines.append("FILTER " + _expression(step.predicate.text))
        elif isinstance(step, Let):
            lines.append("LET " + ", ".join(_item(item) for item in step.bindings))
        elif isinstance(step, Keep):
            lines.append("KEEP " + ", ".join(_item(item) for item in step.items))
        elif isinstance(step, Aggregate):
            value = "AGG " + ", ".join(
                f"{_expression(item.call.text)} AS {item.alias}"
                for item in step.bindings
            )
            if step.per:
                value += " PER " + ", ".join(
                    (
                        f"{_expression(item.expression.text)} AS {item.alias}"
                        if item.alias
                        else _expression(item.expression.text)
                    )
                    for item in step.per
                )
            lines.append(value)
            if step.having:
                lines.append("HAVING " + _expression(step.having.text))
        elif isinstance(step, Dedup):
            lines.append("DEDUP BY " + ", ".join(_expression(key.text) for key in step.keys))
        elif isinstance(step, Expand):
            lines.append(f"EXPAND {_expression(step.expression.text)} AS {step.variable}")
        elif isinstance(step, OptionalBlock):
            lines.append("OPTIONAL {")
            lines.extend(_indent(_steps(step.steps), 2))
            lines.append("}")
        elif isinstance(step, Scope):
            lines.append("SCOPE {")
            lines.extend(_indent(_steps(step.steps), 2))
            lines.append("} EXPOSE " + ", ".join(step.expose))
        elif isinstance(step, AnyClause):
            lines.append("ANY ALL" if step.all else "ANY")
            for index, branch in enumerate(step.branches):
                if index:
                    lines.append("OR")
                lines.append("  SCOPE {")
                lines.extend(_indent(_steps(branch.steps), 4))
                lines.append("  }")
            lines.append("EXPOSE " + ", ".join(step.expose))
        elif isinstance(step, Order):
            lines.append("ORDER BY " + _order_items(step.items))
        elif isinstance(step, Top):
            lines.append("ORDER BY " + _order_items(step.items))
            if step.offset is not None:
                lines.append(f"OFFSET {step.offset}")
            lines.append(f"LIMIT {step.limit}")
        elif isinstance(step, Offset):
            lines.append(f"OFFSET {step.offset}")
        elif isinstance(step, Take):
            lines.append(f"LIMIT {step.limit}")
        else:
            raise TypeError(f"unsupported AST step: {type(step).__name__}")
    return lines


def serialize_query(query: Query) -> str:
    """Serialize one AST using only the SIRC v0.3 canonical vocabulary."""
    answer = "ANSWER "
    if query.answer.distinct:
        answer += "DISTINCT "
    answer += ", ".join(_item(item) for item in query.answer.items)
    return "\n".join([answer, *_steps(query.steps)])


def canonicalize_sirc(text: str) -> str:
    """Migrate accepted v0.2 syntax to the unique v0.3 representation."""
    return serialize_query(parse_sirc(text))


canonicalize_natcypher = canonicalize_sirc
