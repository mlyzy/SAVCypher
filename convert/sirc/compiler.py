from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

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
    OrderItem,
    Pattern,
    Query,
    Relationship,
    Scope,
    Step,
    Take,
    Top,
)
from sirc.parser import parse_sirc


@dataclass
class CompilationResult:
    cypher: str = ""
    supported: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SIRCCompileError(ValueError):
    pass


NatCypherCompileError = SIRCCompileError


def _expression(text: str) -> str:
    """Convert canonical JSON-style strings without rewriting identifiers."""
    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != '"':
            output.append(text[index])
            index += 1
            continue
        index += 1
        value: list[str] = []
        while index < len(text):
            char = text[index]
            if char == '"' and (index == 0 or text[index - 1] != "\\"):
                index += 1
                break
            if char == "\\" and index + 1 < len(text):
                index += 1
                escapes = {"n": "\n", "r": "\r", "t": "\t"}
                value.append(escapes.get(text[index], text[index]))
                index += 1
            else:
                value.append(char)
                index += 1
        output.append("'" + "".join(value).replace("\\", "\\\\").replace("'", "\\'") + "'")
    return "".join(output)


def _mapped_expression(text: str, aliases: dict[str, str]) -> str:
    rendered = text
    for expression, alias in sorted(aliases.items(), key=lambda item: -len(item[0])):
        rendered = re.sub(
            rf"(?<![\w.]){re.escape(expression)}(?![\w.])", alias, rendered
        )
    return _expression(rendered)


def _properties(properties) -> str:
    if not properties:
        return ""
    return " {" + ", ".join(f"{name}: {_expression(value.text)}" for name, value in properties) + "}"


def _node(node: Node) -> str:
    variable = node.variable or ""
    labels = "".join(f":{label}" for label in node.labels)
    return f"({variable}{labels}{_properties(node.properties)})"


def _relationship(rel: Relationship) -> str:
    body = rel.variable or ""
    if rel.types:
        body += ":" + "|".join(rel.types)
    body += rel.hop or ""
    body += _properties(rel.properties)
    if rel.direction == "out":
        return f"-[{body}]->"
    if rel.direction == "in":
        return f"<-[{body}]-"
    return f"-[{body}]-"


def _pattern(pattern: Pattern) -> str:
    chain = _node(pattern.nodes[0])
    for relation, node in zip(pattern.relationships, pattern.nodes[1:]):
        chain += _relationship(relation) + _node(node)
    if pattern.path_mode == "SHORTEST":
        chain = f"shortestPath({chain})"
    elif pattern.path_mode == "ALL_SHORTEST":
        chain = f"allShortestPaths({chain})"
    if pattern.path_variable:
        chain = f"{pattern.path_variable} = {chain}"
    return chain


def _order(items: tuple[OrderItem, ...], aliases: Optional[dict[str, str]] = None) -> str:
    aliases = aliases or {}
    return ", ".join(
        f"{_mapped_expression(item.expression.text, aliases)} {item.direction}"
        for item in items
    )


def _result_item(item, aliases: Optional[dict[str, str]] = None) -> str:
    rendered = _mapped_expression(item.expression.text, aliases or {})
    return f"{rendered} AS {item.alias}" if item.alias else rendered


def _compile_optional(block: OptionalBlock) -> list[str]:
    graphs = [step for step in block.steps if isinstance(step, Graph)]
    if not graphs:
        raise SIRCCompileError("OPTIONAL block requires GRAPH")
    if any(not isinstance(step, (Graph, Filter, OptionalBlock)) for step in block.steps):
        raise SIRCCompileError("unsupported step inside OPTIONAL")
    patterns = [_pattern(pattern) for graph in graphs for pattern in graph.patterns]
    lines = ["OPTIONAL MATCH " + ", ".join(patterns)]
    filters = [step for step in block.steps if isinstance(step, Filter)]
    if filters:
        lines.append("WHERE " + " AND ".join(f"({_expression(item.predicate.text)})" for item in filters))
    for nested in (step for step in block.steps if isinstance(step, OptionalBlock)):
        lines.extend(_compile_optional(nested))
    return lines


def _compile_any(step: AnyClause) -> list[str]:
    if len(step.branches) < 2:
        raise SIRCCompileError("ANY requires at least two branches")
    branches = []
    for branch in step.branches:
        body = _compile_steps(branch.steps)
        body.append("RETURN " + ", ".join(step.expose))
        branches.append("\n  ".join(body))
    union = "\n  UNION ALL\n  " if step.all else "\n  UNION\n  "
    return ["CALL {\n  " + union.join(branches) + "\n}"]


def _compile_steps(
    steps: tuple[Step, ...], aliases: Optional[dict[str, str]] = None
) -> list[str]:
    if aliases is None:
        aliases = {}
    lines: list[str] = []
    for step_index, step in enumerate(steps):
        if isinstance(step, Graph):
            lines.append("MATCH " + ", ".join(_pattern(pattern) for pattern in step.patterns))
        elif isinstance(step, OptionalBlock):
            lines.extend(_compile_optional(step))
        elif isinstance(step, Filter):
            lines.append("WHERE " + _mapped_expression(step.predicate.text, aliases))
        elif isinstance(step, Let):
            next_step = steps[step_index + 1] if step_index + 1 < len(steps) else None
            if isinstance(next_step, Aggregate) and all(item.alias for item in step.bindings):
                for item in step.bindings:
                    aliases[item.alias] = _mapped_expression(item.expression.text, aliases)
            else:
                lines.append("WITH *, " + ", ".join(_result_item(item, aliases) for item in step.bindings))
        elif isinstance(step, Keep):
            lines.append("WITH " + ", ".join(_result_item(item, aliases) for item in step.items))
        elif isinstance(step, Dedup):
            lines.append("WITH DISTINCT " + ", ".join(_mapped_expression(key.text, aliases) for key in step.keys))
        elif isinstance(step, Aggregate):
            projections = []
            next_aliases: dict[str, str] = {}
            for ordinal, item in enumerate(step.per):
                expression = item.expression.text
                rendered = _mapped_expression(expression, aliases)
                if item.alias:
                    projections.append(f"{rendered} AS {item.alias}")
                elif expression in aliases:
                    projections.append(f"{rendered} AS {expression}")
                elif re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", expression):
                    generated = f"__group{ordinal}"
                    projections.append(f"{rendered} AS {generated}")
                    next_aliases[expression] = generated
                else:
                    projections.append(rendered)
            projections.extend(
                f"{_mapped_expression(item.call.text, aliases)} AS {item.alias}"
                for item in step.bindings
            )
            lines.append("WITH " + ", ".join(projections))
            aliases.clear()
            aliases.update(next_aliases)
            if step.having:
                lines.append("WHERE " + _mapped_expression(step.having.text, aliases))
        elif isinstance(step, Expand):
            lines.append(f"UNWIND {_expression(step.expression.text)} AS {step.variable}")
        elif isinstance(step, Scope):
            lines.extend(_compile_steps(step.steps, aliases))
            lines.append("WITH " + ", ".join(step.expose))
            aliases.clear()
        elif isinstance(step, AnyClause):
            lines.extend(_compile_any(step))
        elif isinstance(step, Order):
            lines.append("ORDER BY " + _order(step.items, aliases))
        elif isinstance(step, Top):
            lines.append("ORDER BY " + _order(step.items, aliases))
            if step.offset is not None:
                lines.append(f"SKIP {step.offset}")
            lines.append(f"LIMIT {step.limit}")
        elif isinstance(step, Offset):
            lines.append(f"SKIP {step.offset}")
        elif isinstance(step, Take):
            lines.append(f"LIMIT {step.limit}")
        else:
            raise SIRCCompileError(f"unsupported AST step: {type(step).__name__}")
    return lines


def compile_query(query: Query) -> str:
    """Compile an already parsed and validated query."""
    trailing = []
    body = list(query.steps)
    while body and isinstance(body[-1], (Order, Top, Offset, Take)):
        trailing.insert(0, body.pop())
    aliases: dict[str, str] = {}
    lines = _compile_steps(tuple(body), aliases)
    lines.append(
        "RETURN "
        + ("DISTINCT " if query.answer.distinct else "")
        + ", ".join(_result_item(item, aliases) for item in query.answer.items)
    )
    lines.extend(_compile_steps(tuple(trailing), aliases))
    return "\n".join(lines)


def compile_sirc(
    sirc: str,
    schema: Optional[str] = None,
    *,
    validate: bool = True,
) -> CompilationResult:
    """Parse, optionally schema-validate, and compile SIRC."""
    result = CompilationResult()
    try:
        query = parse_sirc(sirc)
        if validate:
            from sirc.validator import validate_query

            validation = validate_query(query, schema)
            if not validation.valid:
                result.supported = False
                result.errors.extend(validation.errors)
                return result
            result.warnings.extend(validation.warnings)
        result.cypher = compile_query(query)
    except (ValueError, SIRCCompileError) as exc:
        result.supported = False
        result.errors.append(f"{type(exc).__name__}: {exc}")
    return result


compile_natcypher = compile_sirc
