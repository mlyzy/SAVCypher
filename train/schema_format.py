from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_FORMAT_VERSION = "compact-v1"
_STRUCTURAL_KEYS = {"node_props", "rel_props", "relationships"}
_IGNORED_KEYS = {
    "metadata",
    "examples",
    "example",
    "distinct_count",
    "distinctValues",
    "min",
    "max",
    "count",
    "size",
    "statistics",
    "stats",
}


def _clean_text(value: str) -> str:
    lines = [
        line.rstrip()
        for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _question_matches(value: Any, question: str) -> bool:
    if not question:
        return False
    rendered = str(value).strip()
    return bool(
        rendered
        and re.search(
            rf"(?<![\w]){re.escape(rendered)}(?![\w])",
            question,
            re.IGNORECASE,
        )
    )


def _property_text(item: Any, question: str) -> str:
    if not isinstance(item, Mapping):
        return str(item)
    name = str(item.get("property", item.get("name", "<unnamed>")))
    data_type = str(item.get("type", "UNKNOWN"))
    text = f"{name}:{data_type}"
    values = item.get("values")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        matched = [value for value in values if _question_matches(value, question)]
        if matched:
            text += "=" + ",".join(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                for value in matched
            )
    return text


def _format_property_groups(title: str, groups: Any, question: str) -> list[str]:
    if not isinstance(groups, Mapping) or not groups:
        return []
    lines = [title]
    for label in sorted(groups, key=str):
        properties = groups[label]
        if not isinstance(properties, Sequence) or isinstance(properties, (str, bytes)):
            properties = [properties]
        rendered = sorted(
            (_property_text(item, question) for item in properties), key=str
        )
        lines.append(f"  {label}" + "{" + ",".join(rendered) + "}")
    return lines


def _strip_nonstructural(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_nonstructural(item)
            for key, item in value.items()
            if key not in _IGNORED_KEYS and key != "values"
        }
    if isinstance(value, list):
        return [_strip_nonstructural(item) for item in value]
    return value


def format_schema(schema: Any, question: str = "") -> str:
    """Return an inference-compatible compact schema.

    Relationships are first and never pruned. All labels, relationship types,
    property names, and property types are retained. Statistics and examples
    are removed, except values occurring literally in the question.
    """
    if isinstance(schema, str):
        text = _clean_text(schema)
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        if isinstance(parsed, (Mapping, list)):
            return format_schema(parsed, question=question)
        return text

    if isinstance(schema, Mapping):
        sections: list[list[str]] = []
        if "relationships" in schema:
            relationships = schema["relationships"]
            if not isinstance(relationships, Sequence) or isinstance(
                relationships, (str, bytes)
            ):
                relationships = [relationships]
            grouped_relationships: dict[str, set[tuple[str, str]]] = {}
            rendered_other = []
            for relationship in relationships:
                if isinstance(relationship, Mapping) and {
                    "start",
                    "type",
                    "end",
                }.issubset(relationship):
                    grouped_relationships.setdefault(
                        str(relationship["type"]), set()
                    ).add(
                        (str(relationship["start"]), str(relationship["end"]))
                    )
                else:
                    rendered_other.append(
                        json.dumps(relationship, ensure_ascii=False, sort_keys=True)
                    )
            rendered = [
                f"{rel_type}:"
                + "|".join(
                    f"{start}->{end}"
                    for start, end in sorted(grouped_relationships[rel_type])
                )
                for rel_type in sorted(grouped_relationships)
            ]
            rendered.extend(sorted(rendered_other))
            sections.append(
                ["Relationships:"]
                + ([f"  {item}" for item in sorted(rendered)] or ["  (none)"])
            )
        node_lines = _format_property_groups(
            "Node properties:", schema.get("node_props"), question
        )
        if node_lines:
            sections.append(node_lines)
        rel_lines = _format_property_groups(
            "Relationship properties:", schema.get("rel_props"), question
        )
        if rel_lines:
            sections.append(rel_lines)

        remainder = {
            key: _strip_nonstructural(value)
            for key, value in schema.items()
            if key not in _STRUCTURAL_KEYS and key not in _IGNORED_KEYS
        }
        if remainder:
            sections.append(
                [
                    "Additional schema:"
                    + json.dumps(remainder, ensure_ascii=False, sort_keys=True)
                ]
            )
        return "\n\n".join("\n".join(section) for section in sections)

    if isinstance(schema, list):
        return json.dumps(
            _strip_nonstructural(schema), ensure_ascii=False, sort_keys=True
        )
    return ""


def schema_symbols(schema: Any) -> dict[str, set[str]]:
    """Extract critical names which the compact representation must preserve."""
    if isinstance(schema, str):
        try:
            schema = json.loads(_clean_text(schema))
        except (json.JSONDecodeError, TypeError):
            return {
                "labels": set(),
                "relationships": set(),
                "properties": set(),
                "types": set(),
            }
    result = {
        "labels": set(),
        "relationships": set(),
        "properties": set(),
        "types": set(),
    }
    if not isinstance(schema, Mapping):
        return result
    node_props = schema.get("node_props", {})
    if isinstance(node_props, Mapping):
        result["labels"].update(str(label) for label in node_props)
        _collect_property_symbols(
            node_props.values(), result["properties"], result["types"]
        )
    rel_props = schema.get("rel_props", {})
    if isinstance(rel_props, Mapping):
        result["relationships"].update(str(rel_type) for rel_type in rel_props)
        _collect_property_symbols(
            rel_props.values(), result["properties"], result["types"]
        )
    relationships = schema.get("relationships", [])
    if isinstance(relationships, Mapping) or isinstance(relationships, (str, bytes)):
        relationships = [relationships]
    if isinstance(relationships, Sequence):
        for relationship in relationships:
            if isinstance(relationship, Mapping):
                for key in ("start", "end"):
                    if key in relationship:
                        result["labels"].add(str(relationship[key]))
                if "type" in relationship:
                    result["relationships"].add(str(relationship["type"]))
    return result


def _collect_property_symbols(
    groups: Iterable[Any], names: set[str], types: set[str]
) -> None:
    for properties in groups:
        if isinstance(properties, Mapping) or isinstance(properties, (str, bytes)):
            properties = [properties]
        if not isinstance(properties, Sequence):
            continue
        for item in properties:
            if isinstance(item, Mapping):
                name = item.get("property", item.get("name"))
                if name is not None:
                    names.add(str(name))
                if "type" in item:
                    types.add(str(item["type"]))


def audit_schema_symbols(
    schema: Any, formatted: str, cypher: str = ""
) -> dict[str, Any]:
    """Audit complete and gold-query-referenced critical symbol coverage."""
    symbols = schema_symbols(schema)
    missing = {
        category: sorted(name for name in names if name not in formatted)
        for category, names in symbols.items()
    }
    referenced = {
        category: {
            name
            for name in names
            if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", cypher)
        }
        for category, names in symbols.items()
    }
    referenced_missing = {
        category: sorted(name for name in names if name not in formatted)
        for category, names in referenced.items()
    }
    total = sum(map(len, symbols.values()))
    covered = total - sum(len(names) for names in missing.values())
    referenced_total = sum(map(len, referenced.values()))
    referenced_covered = referenced_total - sum(
        len(names) for names in referenced_missing.values()
    )
    return {
        "total": total,
        "covered": covered,
        "coverage": covered / total if total else 1.0,
        "missing": missing,
        "referenced_total": referenced_total,
        "referenced_covered": referenced_covered,
        "referenced_coverage": (
            referenced_covered / referenced_total if referenced_total else 1.0
        ),
        "referenced_missing": referenced_missing,
    }


def estimate_tokens(text: str) -> int:
    """Return a conservative tokenizer-independent manifest token estimate."""
    return sum(
        max(1, math.ceil(len(piece.encode("utf-8")) / 4))
        for piece in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    )
