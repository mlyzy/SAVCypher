from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    Offset,
    OptionalBlock,
    Order,
    Query,
    Scope,
    Take,
    Top,
)


@dataclass(frozen=True)
class Binding:
    kind: str
    labels: frozenset[str] = frozenset()
    rel_types: frozenset[str] = frozenset()
    nullable: bool = False
    element: Optional["Binding"] = None


@dataclass
class CompactSchema:
    labels: set[str] = field(default_factory=set)
    node_properties: dict[str, set[str]] = field(default_factory=dict)
    relationships: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    relationship_properties: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def reject(self, message: str) -> None:
        self.valid = False
        if message not in self.errors:
            self.errors.append(message)


def parse_compact_schema(text: Optional[str]) -> CompactSchema:
    """Parse compact-v1, including ``TYPE:A->B|C->D`` relationships."""
    schema = CompactSchema()
    if not text or not str(text).strip():
        return schema
    section = ""
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and line in {
            "Relationships:",
            "Node properties:",
            "Relationship properties:",
        }:
            section = line
            continue
        if line == "(none)":
            continue
        if section == "Relationships:":
            match = re.match(r"^([^:]+):(.+)$", line)
            if not match:
                continue
            rel_type = match.group(1).strip()
            pairs = set()
            for pair in match.group(2).split("|"):
                if "->" not in pair:
                    continue
                start, end = pair.split("->", 1)
                start, end = start.strip(), end.strip()
                pairs.add((start, end))
                schema.labels.update((start, end))
            schema.relationships.setdefault(rel_type, set()).update(pairs)
        elif section in {"Node properties:", "Relationship properties:"}:
            match = re.match(r"^([^{}]+)\{(.*)\}$", line)
            if not match:
                continue
            owner, body = match.group(1).strip(), match.group(2)
            properties = {
                item.group(1)
                for item in re.finditer(r"(?:^|,)([A-Za-z_]\w*)\s*:", body)
            }
            if section == "Node properties:":
                schema.labels.add(owner)
                schema.node_properties.setdefault(owner, set()).update(properties)
            else:
                schema.relationship_properties.setdefault(owner, set()).update(properties)
    return schema


class _Validator:
    def __init__(self, schema: CompactSchema):
        self.schema = schema
        self.result = ValidationResult()
        self.has_source = False

    def reject(self, message: str) -> None:
        self.result.reject(message)

    def expression(self, text: str, env: dict[str, Binding], context: str) -> None:
        scrubbed = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", text)
        scrubbed = re.sub(r"//.*$", "", scrubbed).strip()
        local_variables = set(re.findall(r"\b([A-Za-z_]\w*)\s+IN\b", scrubbed, re.I))
        local_variables.update(
            re.findall(r"(?i)\bREDUCE\s*\(\s*([A-Za-z_]\w*)\s*=", scrubbed)
        )
        local_variables.update(re.findall(r"[\(\[]\s*([A-Za-z_]\w*)\s*(?::|\))", scrubbed))
        local_variables.update(re.findall(r"\bAS\s+([A-Za-z_]\w*)", scrubbed, re.I))
        namespaces = set(
            re.findall(r"\b([A-Za-z_]\w*)(?:\.[A-Za-z_]\w*)+\s*\(", scrubbed)
        )
        map_keys = set(re.findall(r"(?:\{|,)\s*([A-Za-z_]\w*)\s*:", scrubbed))
        for variable, prop in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", scrubbed):
            if variable in namespaces:
                continue
            if variable in local_variables:
                continue
            binding = env.get(variable)
            if binding is None:
                self.reject(f"undefined_variable:{context}:{variable}")
                continue
            if binding.kind == "node":
                if not binding.labels:
                    self.reject(f"ambiguous_unlabeled_property:{variable}.{prop}")
                elif any(prop not in self.schema.node_properties.get(label, set()) for label in binding.labels):
                    self.reject(f"unknown_node_property:{variable}.{prop}")
            elif binding.kind == "relationship":
                if not binding.rel_types:
                    self.reject(f"ambiguous_untyped_relationship_property:{variable}.{prop}")
                elif any(prop not in self.schema.relationship_properties.get(rel_type, set()) for rel_type in binding.rel_types):
                    self.reject(f"unknown_relationship_property:{variable}.{prop}")
            elif binding.kind not in {"temporal", "map", "point", "dynamic"}:
                self.reject(f"property_on_non_graph_value:{variable}.{prop}")
        known_functions = {
            "COUNT", "SUM", "AVG", "MIN", "MAX", "COLLECT", "SIZE", "LENGTH",
            "NODES", "RELATIONSHIPS", "TYPE", "LABELS", "TO_FLOAT", "TO_INTEGER",
            "TO_STRING", "TOFLOAT", "TOINTEGER", "TOSTRING", "COALESCE",
            "HEAD", "LAST", "DATE", "DATETIME",
            "LOCALDATETIME", "TIME", "DURATION", "ROUND", "ABS", "LOWER", "UPPER",
            "TRIM", "SPLIT", "REPLACE", "SUBSTRING", "DISTANCE", "POINT",
            "ID", "TOLOWER", "TOUPPER", "YEAR", "PERCENTILECONT", "PERCENTILEDISC",
            "ANY", "ALL", "NONE", "SINGLE", "REDUCE", "RANGE", "EXISTS",
        }
        non_function_words = {
            "NOT", "MATCH", "EXISTS", "AND", "OR", "DISTINCT", "WHEN", "CASE",
        }
        for function in re.findall(r"(?<!\.)\b([A-Za-z_]\w*)\s*\(", scrubbed):
            if function.upper() not in known_functions | non_function_words:
                self.reject(f"unsupported_function:{function}")
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", scrubbed))
        keywords = {
            "AS", "ASC", "DESC", "AND", "OR", "NOT", "IS", "NULL", "TRUE", "FALSE",
            "IN", "CONTAINS", "STARTS", "ENDS", "WITH", "DISTINCT", "CASE", "WHEN",
            "THEN", "ELSE", "END", "WHERE", "MATCH", "UNION", "ALL", "OVER",
            "ORDER", "BY",
        } | known_functions
        keywords = {item.upper() for item in keywords}
        property_names = set(re.findall(r"\.\s*([A-Za-z_]\w*)", scrubbed))
        schema_symbols = (
            self.schema.labels
            | set(self.schema.relationships)
            | {prop for props in self.schema.node_properties.values() for prop in props}
            | {prop for props in self.schema.relationship_properties.values() for prop in props}
        )
        parameter_names = set(re.findall(r"\$([A-Za-z_]\w*)", scrubbed))
        for token in tokens - property_names - local_variables - namespaces - map_keys - parameter_names:
            if (
                token.upper() not in keywords
                and token not in schema_symbols
                and token not in env
                and not re.search(rf"(?:\b|\.){re.escape(token)}\s*\(", scrubbed)
            ):
                self.reject(f"undefined_variable:{context}:{token}")

    def binding_for_expression(self, text: str, env: dict[str, Binding]) -> Binding:
        value = re.sub(r"//.*$", "", text).strip()
        if value in env:
            return env[value]
        indexed = re.match(r"^([A-Za-z_]\w*)\s*\[", value)
        if indexed and indexed.group(1) in env:
            source = env[indexed.group(1)]
            return source.element or Binding("dynamic")
        collected_index = re.match(
            r"(?is)^COLLECT\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_]\w*)\s*\)\s*\[",
            value,
        )
        if collected_index and collected_index.group(1) in env:
            return env[collected_index.group(1)]
        collected = re.match(r"(?is)^COLLECT\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_]\w*)\s*\)", value)
        if collected and collected.group(1) in env:
            return Binding("list", element=env[collected.group(1)])
        if value.startswith("["):
            list_source = re.search(r"\|\s*([A-Za-z_]\w*)\s*$", value)
            element = env.get(list_source.group(1)) if list_source else None
            return Binding("list", element=element or Binding("dynamic"))
        if re.match(r"(?i)^(DATE|DATETIME|LOCALDATETIME|TIME|DURATION)\s*\(", value):
            return Binding("temporal")
        if re.match(r"(?i)^POINT\s*\(", value):
            return Binding("point")
        if value.startswith("{"):
            return Binding("map")
        return Binding("scalar")

    def graph(self, graph: Graph, env: dict[str, Binding], optional: bool = False) -> None:
        self.has_source = True
        # Bind all explicit labels first so repeated variables constrain every pattern.
        for pattern in graph.patterns:
            if pattern.path_variable:
                if pattern.path_variable in env:
                    self.reject(f"variable_rebinding:{pattern.path_variable}")
                env[pattern.path_variable] = Binding("path", nullable=optional)
            for node in pattern.nodes:
                for label in node.labels:
                    if label not in self.schema.labels:
                        self.reject(f"unknown_label:{label}")
                if node.variable:
                    previous = env.get(node.variable)
                    if previous and previous.kind != "node":
                        self.reject(f"variable_type_conflict:{node.variable}")
                    labels = frozenset(node.labels) | (previous.labels if previous else frozenset())
                    env[node.variable] = Binding("node", labels, nullable=optional or bool(previous and previous.nullable))
            for rel in pattern.relationships:
                for rel_type in rel.types:
                    if rel_type not in self.schema.relationships:
                        self.reject(f"unknown_relationship:{rel_type}")
                if not rel.types:
                    self.reject("relationship_type_required")
                if rel.variable:
                    previous = env.get(rel.variable)
                    if previous:
                        self.reject(f"variable_rebinding:{rel.variable}")
                    env[rel.variable] = Binding("relationship", rel_types=frozenset(rel.types), nullable=optional)

        # Validate topology and infer a label only when the schema proves one.
        for pattern in graph.patterns:
            for left_node, rel, right_node in zip(pattern.nodes, pattern.relationships, pattern.nodes[1:]):
                left = env.get(left_node.variable) if left_node.variable else Binding("node", frozenset(left_node.labels))
                right = env.get(right_node.variable) if right_node.variable else Binding("node", frozenset(right_node.labels))
                candidates = set()
                for rel_type in rel.types:
                    candidates.update((rel_type, start, end) for start, end in self.schema.relationships.get(rel_type, set()))
                compatible = []
                for rel_type, start, end in candidates:
                    expected_left, expected_right = (end, start) if rel.direction == "in" else (start, end)
                    if rel.direction == "undirected":
                        direct = (start, end)
                        reverse = (end, start)
                        for expected_left, expected_right in (direct, reverse):
                            if self._labels_fit(left.labels, expected_left) and self._labels_fit(right.labels, expected_right):
                                compatible.append((rel_type, expected_left, expected_right))
                    elif self._labels_fit(left.labels, expected_left) and self._labels_fit(right.labels, expected_right):
                        compatible.append((rel_type, expected_left, expected_right))
                if not compatible:
                    self.reject(f"relationship_endpoint_mismatch:{'|'.join(rel.types)}")
                    continue
                left_labels = {item[1] for item in compatible}
                right_labels = {item[2] for item in compatible}
                if not left.labels and len(left_labels) == 1 and left_node.variable:
                    env[left_node.variable] = replace(left, labels=frozenset(left_labels))
                    left = env[left_node.variable]
                if not right.labels and len(right_labels) == 1 and right_node.variable:
                    env[right_node.variable] = replace(right, labels=frozenset(right_labels))

        for pattern in graph.patterns:
            for node in pattern.nodes:
                binding = env.get(node.variable) if node.variable else Binding("node", frozenset(node.labels))
                for prop, expression in node.properties:
                    if not binding.labels:
                        self.reject(f"ambiguous_unlabeled_property:{node.variable or 'anonymous'}.{prop}")
                    elif any(prop not in self.schema.node_properties.get(label, set()) for label in binding.labels):
                        self.reject(f"unknown_node_property:{node.variable or 'anonymous'}.{prop}")
                    self.expression(expression.text, env, f"property_map:{prop}")
            for rel in pattern.relationships:
                for prop, expression in rel.properties:
                    if any(prop not in self.schema.relationship_properties.get(rel_type, set()) for rel_type in rel.types):
                        self.reject(f"unknown_relationship_property:{rel.variable or 'anonymous'}.{prop}")
                    self.expression(expression.text, env, f"relationship_map:{prop}")

    @staticmethod
    def _labels_fit(labels: frozenset[str], expected: str) -> bool:
        return not labels or expected in labels

    def steps(self, steps, env: dict[str, Binding]) -> dict[str, Binding]:
        for step in steps:
            if isinstance(step, Graph):
                self.graph(step, env)
            elif isinstance(step, Filter):
                self.expression(step.predicate.text, env, "FILTER")
            elif isinstance(step, OptionalBlock):
                inner = dict(env)
                self.steps(step.steps, inner)
                for name, binding in inner.items():
                    if name not in env:
                        env[name] = replace(binding, nullable=True)
            elif isinstance(step, Let):
                for item in step.bindings:
                    self.expression(item.expression.text, env, "LET")
                    if not item.alias:
                        self.reject("LET_alias_required")
                    elif item.alias in env:
                        self.reject(f"variable_rebinding:{item.alias}")
                    else:
                        env[item.alias] = self.binding_for_expression(item.expression.text, env)
            elif isinstance(step, Keep):
                new_env: dict[str, Binding] = {}
                for item in step.items:
                    self.expression(item.expression.text, env, "KEEP")
                    bare = re.fullmatch(r"[A-Za-z_]\w*", item.expression.text)
                    name = item.alias or (bare.group(0) if bare else None)
                    if not name:
                        self.reject(f"KEEP_expression_requires_alias:{item.expression.text}")
                    elif item.alias and item.alias in env:
                        self.reject(f"variable_rebinding:{item.alias}")
                    else:
                        new_env[name] = (
                            env[name]
                            if bare and name in env and not item.alias
                            else self.binding_for_expression(item.expression.text, env)
                        )
                env = new_env
            elif isinstance(step, Dedup):
                new_env = {}
                for key in step.keys:
                    self.expression(key.text, env, "DEDUP")
                    if key.text in env:
                        new_env[key.text] = env[key.text]
                env = new_env
            elif isinstance(step, Aggregate):
                new_env = {}
                for key in step.per:
                    expression = key.expression
                    self.expression(expression.text, env, "PER")
                    name = key.alias or expression.text
                    if key.alias:
                        new_env[name] = self.binding_for_expression(expression.text, env)
                    elif expression.text in env:
                        new_env[name] = env[expression.text]
                    elif re.fullmatch(r"([A-Za-z_]\w*)\.[A-Za-z_]\w*", expression.text):
                        root = expression.text.split(".", 1)[0]
                        if root in env:
                            new_env[root] = env[root]
                    else:
                        self.reject(f"PER_key_must_be_bound_variable:{expression.text}")
                for binding in step.bindings:
                    self.expression(binding.call.text, env, "AGG")
                    if binding.alias in env and binding.alias not in new_env:
                        self.reject(f"variable_rebinding:{binding.alias}")
                    new_env[binding.alias] = self.binding_for_expression(binding.call.text, env)
                env = new_env
                if step.having:
                    self.expression(step.having.text, env, "HAVING")
            elif isinstance(step, Expand):
                self.expression(step.expression.text, env, "EXPAND")
                if step.variable in env:
                    self.reject(f"variable_rebinding:{step.variable}")
                source = self.binding_for_expression(step.expression.text, env)
                env[step.variable] = source.element or Binding("dynamic")
            elif isinstance(step, Scope):
                inner = self.steps(step.steps, dict(env))
                for name in step.expose:
                    if name not in inner:
                        self.reject(f"undefined_expose:{name}")
                env = {name: inner[name] for name in step.expose if name in inner}
            elif isinstance(step, AnyClause):
                branch_bindings = []
                for branch in step.branches:
                    inner = self.steps(branch.steps, dict(env))
                    branch_bindings.append(inner)
                    for name in step.expose:
                        if name not in inner:
                            self.reject(f"undefined_expose:{name}")
                for name in step.expose:
                    kinds = {inner[name].kind for inner in branch_bindings if name in inner}
                    if len(kinds) > 1:
                        self.reject(f"any_branch_type_mismatch:{name}")
                    if kinds:
                        env[name] = next(inner[name] for inner in branch_bindings if name in inner)
            elif isinstance(step, (Order, Top)):
                for item in step.items:
                    self.expression(item.expression.text, env, type(step).__name__.upper())
        return env

    def query(self, query: Query) -> ValidationResult:
        trailing = []
        body = list(query.steps)
        while body and isinstance(body[-1], (Order, Top, Offset, Take)):
            trailing.insert(0, body.pop())
        env = self.steps(tuple(body), {})
        for item in query.answer.items:
            self.expression(item.expression.text, env, "ANSWER")
            if item.alias:
                env[item.alias] = Binding("scalar")
        self.steps(tuple(trailing), env)
        if not self.has_source:
            self.reject("query_has_no_graph_source")
        return self.result


def validate_query(query: Query, schema: Optional[str]) -> ValidationResult:
    parsed = parse_compact_schema(schema)
    validator = _Validator(parsed)
    if not schema or not str(schema).strip():
        validator.reject("compact_schema_required")
        return validator.result
    return validator.query(query)
