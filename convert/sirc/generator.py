from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CLAUSE_KEYWORDS = (
    "OPTIONAL MATCH",
    "ORDER BY",
    "UNION ALL",
    "MATCH",
    "WHERE",
    "WITH",
    "UNWIND",
    "RETURN",
    "SKIP",
    "LIMIT",
    "UNION",
    "CALL",
)
AGGREGATE_RE = re.compile(
    r"(?is)^\s*(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*"
    r"\(\s*(DISTINCT\s+)?(.+?)\s*\)\s*"
    r"(?:AS\s+(`[^`]+`|[A-Za-z_]\w*))?\s*$"
)
ALIAS_RE = re.compile(
    r"(?is)^\s*(.+?)\s+AS\s+(`[^`]+`|[A-Za-z_]\w*)\s*$"
)
IDENTIFIER_RE = re.compile(r"^(?:`[^`]+`|[A-Za-z_]\w*)$")


@dataclass
class ConversionResult:
    sirc: str = ""
    supported: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def reject(self, message: str) -> None:
        self.supported = False
        if message not in self.errors:
            self.errors.append(message)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def natcypher(self) -> str:
        return self.sirc

    @natcypher.setter
    def natcypher(self, value: str) -> None:
        self.sirc = value


@dataclass
class _Aggregate:
    function: str
    argument: str
    distinct: bool
    alias: str

    def render(self) -> str:
        if self.argument == "*":
            call = "COUNT(*)"
        else:
            distinct = "DISTINCT " if self.distinct else ""
            call = f"{self.function}({distinct}{canonical_expression(self.argument)})"
        return f"{call} AS {self.alias}"


class SIRCConverter:
    """Convert benchmark-style read-only Cypher into canonical SIRC."""

    def convert(self, cypher: str) -> ConversionResult:
        result = ConversionResult(
            metadata={
                "source_cypher": str(cypher or ""),
                "sirc_version": "0.3",
            }
        )
        text = clean_cypher(cypher)
        if not text:
            result.reject("empty_query")
            return result
        if re.search(
            r"(?i)\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|LOAD\s+CSV|FOREACH)\b",
            text,
        ):
            result.reject("write_or_admin_query")
            return result

        clauses = split_clauses(text)
        if not clauses:
            result.reject("no_supported_clause")
            return result
        if any(keyword.startswith("UNION") for keyword, _ in clauses):
            result.reject("top_level_union_requires_branch_converter")
            return result

        return_indexes = [
            index for index, (keyword, _) in enumerate(clauses) if keyword == "RETURN"
        ]
        if len(return_indexes) != 1:
            result.reject("query_must_have_one_top_level_return")
            return result
        return_index = return_indexes[0]

        answer, return_aggregates = self._convert_return(
            clauses[return_index][1], result
        )
        body: List[str] = []
        intermediate_window: List[Tuple[str, str]] = []
        last_with_expose: List[str] = []
        last_operation = ""
        index = 0
        while index < return_index:
            keyword, clause_body = clauses[index]
            next_clause = clauses[index + 1] if index + 1 < return_index else None
            if intermediate_window and keyword not in {"ORDER BY", "SKIP", "LIMIT"}:
                body = self._place_intermediate_window(
                    body,
                    intermediate_window,
                    last_with_expose,
                    result,
                    terminal=False,
                )
                intermediate_window = []
                last_operation = "WINDOW"

            if keyword == "MATCH":
                rendered = convert_patterns(clause_body, result)
                if rendered:
                    body.append(f"GRAPH {rendered}")
                    last_operation = "GRAPH"
            elif keyword == "OPTIONAL MATCH":
                rendered = convert_patterns(clause_body, result)
                if not rendered:
                    index += 1
                    continue
                optional_lines = [rendered]
                if next_clause and next_clause[0] == "WHERE":
                    optional_lines.append(
                        f"FILTER {canonical_expression(next_clause[1])}"
                    )
                    index += 1
                body.append("OPTIONAL {\n  " + "\n  ".join(optional_lines) + "\n}")
                last_operation = "OPTIONAL"
            elif keyword == "WHERE":
                predicate = canonical_expression(clause_body)
                if last_operation == "AGG" and body:
                    body[-1] += f"\nHAVING {predicate}"
                else:
                    body.append(f"FILTER {predicate}")
                last_operation = "FILTER"
            elif keyword == "WITH":
                lines, operation, last_with_expose = convert_with(clause_body, result)
                body.extend(lines)
                last_operation = operation
            elif keyword == "UNWIND":
                unwind = ALIAS_RE.match(clause_body)
                if not unwind:
                    result.reject(f"unsupported_unwind:{clause_body}")
                else:
                    body.append(
                        "EXPAND "
                        f"{canonical_expression(unwind.group(1))} AS "
                        f"{unquote_identifier(unwind.group(2))}"
                    )
                    last_operation = "EXPAND"
            elif keyword == "CALL":
                any_clause = convert_call_union(clause_body, result)
                if any_clause:
                    body.append(any_clause)
                    last_operation = "ANY"
            elif keyword in {"ORDER BY", "SKIP", "LIMIT"}:
                intermediate_window.append((keyword, clause_body))
            else:
                result.reject(f"unsupported_clause:{keyword}")
            index += 1

        if intermediate_window:
            body = self._place_intermediate_window(
                body,
                intermediate_window,
                last_with_expose,
                result,
                terminal=True,
            )
            last_operation = "KEEP"

        if return_aggregates:
            group_items = answer["aggregate_group_items"]
            aggregate_text = ", ".join(item.render() for item in return_aggregates)
            per = f" PER {', '.join(group_items)}" if group_items else ""
            body.append(f"AGG {aggregate_text}{per}")

        window = self._convert_result_window(clauses[return_index + 1 :], result)
        answer_text = "ANSWER "
        if answer["distinct"]:
            answer_text += "DISTINCT "
        answer_text += ", ".join(answer["items"])
        sirc_lines = [answer_text, *body]
        if window:
            sirc_lines.append(window)
        result.sirc = "\n".join(sirc_lines)
        if result.supported:
            try:
                # Route every generated target through the typed AST so direct
                # generation and v0.2 migration share exactly one v0.3 spelling.
                from sirc.serializer import canonicalize_sirc

                result.sirc = canonicalize_sirc(result.sirc)
            except ValueError as exc:
                result.reject(f"canonicalization_error:{exc}")
                result.sirc = ""
        result.metadata.update(
            {
                "clause_count": len(clauses),
                "pipeline_steps": len(body),
            }
        )
        return result

    def _place_intermediate_window(
        self,
        body: List[str],
        window_clauses: Sequence[Tuple[str, str]],
        expose: Sequence[str],
        result: ConversionResult,
        *,
        terminal: bool,
    ) -> List[str]:
        """Place a WITH-stage window without moving it across final ANSWER."""
        if not expose:
            result.reject("intermediate_window_requires_with_contract")
            return body
        window_text = self._convert_result_window(window_clauses, result)
        if window_text:
            body.extend(window_text.splitlines())
        # A following step already makes the window non-trailing. At the end
        # of the pipeline, KEEP is the compact lifecycle barrier that prevents
        # the compiler from moving the window past ANSWER.
        if terminal:
            body.append("KEEP " + ", ".join(expose))
        return body

    def _convert_return(
        self, body: str, result: ConversionResult
    ) -> Tuple[Dict[str, object], List[_Aggregate]]:
        text = body.strip()
        distinct = bool(re.match(r"(?i)^DISTINCT\b", text))
        if distinct:
            text = re.sub(r"(?i)^DISTINCT\s+", "", text, count=1)

        items: List[str] = []
        aggregate_group_items: List[str] = []
        aggregates: List[_Aggregate] = []
        raw_items = split_top_level(text)
        has_aggregate = any(parse_aggregate(item) for item in raw_items)
        for ordinal, raw_item in enumerate(raw_items):
            aggregate = parse_aggregate(raw_item, f"_agg{ordinal}")
            if aggregate:
                aggregates.append(aggregate)
                items.append(aggregate.alias)
                continue
            alias_match = ALIAS_RE.match(raw_item)
            expression = alias_match.group(1) if alias_match else raw_item
            alias = (
                unquote_identifier(alias_match.group(2)) if alias_match else None
            )
            rendered = canonical_expression(expression)
            if alias:
                if has_aggregate:
                    aggregate_group_items.append(f"{rendered} AS {alias}")
                    rendered = alias
                else:
                    rendered += f" AS {alias}"
            else:
                aggregate_group_items.append(rendered)
            items.append(rendered)
        if not items:
            result.reject("empty_return")
            items = ["*"]
        return (
            {
                "distinct": distinct,
                "items": items,
                "aggregate_group_items": aggregate_group_items,
            },
            aggregates,
        )

    def _convert_result_window(
        self, clauses: Sequence[Tuple[str, str]], result: ConversionResult
    ) -> str:
        order: Optional[str] = None
        skip: Optional[int] = None
        limit: Optional[int] = None
        for keyword, body in clauses:
            if keyword == "ORDER BY":
                if order is not None:
                    result.reject("multiple_order_by")
                order = ", ".join(
                    canonical_order_item(item) for item in split_top_level(body)
                )
            elif keyword == "SKIP":
                skip = parse_nonnegative_integer(body, "SKIP", result)
            elif keyword == "LIMIT":
                limit = parse_nonnegative_integer(body, "LIMIT", result)
            else:
                result.reject(f"unsupported_post_return_clause:{keyword}")

        parts = []
        if order:
            parts.append(f"ORDER BY {order}")
        if skip is not None:
            parts.append(f"OFFSET {skip}")
        if limit is not None:
            parts.append(f"LIMIT {limit}")
        return "\n".join(parts)


def convert_with(
    body: str, result: ConversionResult
) -> Tuple[List[str], str, List[str]]:
    text = body.strip()
    distinct = bool(re.match(r"(?i)^DISTINCT\b", text))
    if distinct:
        text = re.sub(r"(?i)^DISTINCT\s+", "", text, count=1)

    raw_items = split_top_level(text)
    aggregates: List[_Aggregate] = []
    group_items: List[str] = []
    keep_items: List[str] = []
    for ordinal, raw_item in enumerate(raw_items):
        aggregate = parse_aggregate(raw_item, f"_agg{ordinal}")
        if aggregate:
            aggregates.append(aggregate)
            continue
        alias_match = ALIAS_RE.match(raw_item)
        expression = canonical_expression(alias_match.group(1) if alias_match else raw_item)
        alias = unquote_identifier(alias_match.group(2)) if alias_match else None
        keep_items.append(f"{expression} AS {alias}" if alias else expression)
        group_items.append(alias or expression)

    lines: List[str] = []
    if aggregates:
        aggregate_text = ", ".join(item.render() for item in aggregates)
        per = f" PER {', '.join(keep_items)}" if keep_items else ""
        lines.append(f"AGG {aggregate_text}{per}")
        if distinct:
            result.warnings.append("distinct_after_aggregation_is_redundant")
        return lines, "AGG", [*group_items, *(item.alias for item in aggregates)]

    if not keep_items:
        result.reject("empty_with")
        return lines, "WITH", []
    lines.append("KEEP " + ", ".join(keep_items))
    if distinct:
        lines.append("DEDUP BY " + ", ".join(group_items))
        return lines, "DEDUP", group_items
    return lines, "KEEP", group_items


def convert_call_union(body: str, result: ConversionResult) -> str:
    text = body.strip()
    if not (text.startswith("{") and text.endswith("}")):
        result.reject("procedure_call_not_supported")
        return ""
    inner = text[1:-1].strip()
    branches, union_all = split_union_branches(inner)
    if len(branches) < 2:
        result.reject("call_without_union_not_supported")
        return ""

    rendered_branches: List[str] = []
    expose_contract: Optional[List[str]] = None
    for branch in branches:
        clauses = split_clauses(branch)
        if not clauses or clauses[0][0] != "MATCH":
            result.reject("union_branch_must_start_with_match")
            return ""
        return_clauses = [item for item in clauses if item[0] == "RETURN"]
        if len(return_clauses) != 1:
            result.reject("union_branch_requires_one_return")
            return ""
        return_index = clauses.index(return_clauses[0])
        branch_lines = [f"GRAPH {convert_patterns(clauses[0][1], result)}"]
        for keyword, clause_body in clauses[1:return_index]:
            if keyword == "WHERE":
                branch_lines.append(f"FILTER {canonical_expression(clause_body)}")
            else:
                result.reject(f"unsupported_union_branch_clause:{keyword}")
                return ""

        contract: List[str] = []
        alias_bindings: List[str] = []
        return_text = return_clauses[0][1]
        if re.match(r"(?i)^DISTINCT\b", return_text.strip()):
            result.reject("branch_distinct_requires_explicit_scope")
            return ""
        for item in split_top_level(return_text):
            alias_match = ALIAS_RE.match(item)
            expression = canonical_expression(alias_match.group(1) if alias_match else item)
            alias = unquote_identifier(alias_match.group(2)) if alias_match else None
            output_name = alias or expression
            if not IDENTIFIER_RE.match(output_name):
                result.reject("union_expose_must_be_identifier")
                return ""
            contract.append(output_name)
            if alias and expression != alias:
                alias_bindings.append(f"{expression} AS {alias}")
        if expose_contract is None:
            expose_contract = contract
        elif expose_contract != contract:
            result.reject("union_branch_contract_mismatch")
            return ""
        if alias_bindings:
            branch_lines.append("LET " + ", ".join(alias_bindings))
        rendered_branches.append(
            "SCOPE {\n    " + "\n    ".join(branch_lines) + "\n  }"
        )

    prefix = "ANY ALL" if union_all else "ANY"
    joined = "\nOR\n  ".join(rendered_branches)
    return (
        f"{prefix}\n  {joined}\n"
        f"EXPOSE {', '.join(expose_contract or [])}"
    )


def convert_patterns(body: str, result: ConversionResult) -> str:
    rendered = []
    for part in split_top_level(body):
        try:
            rendered.append(convert_pattern(part))
        except ValueError as exc:
            result.reject(f"unsupported_pattern:{exc}")
    return ";\n  ".join(item for item in rendered if item)


def convert_pattern(pattern: str) -> str:
    text = pattern.strip()
    path_match = re.match(
        r"(?is)^(`[^`]+`|[A-Za-z_]\w*)\s*=\s*"
        r"(shortestPath|allShortestPaths)\s*\((.*)\)\s*$",
        text,
    )
    if path_match:
        mode = "SHORTEST" if path_match.group(2).lower() == "shortestpath" else "ALL_SHORTEST"
        return (
            f"PATH {unquote_identifier(path_match.group(1))} = {mode} "
            f"{convert_pattern_chain(path_match.group(3))}"
        )
    plain_path = re.match(
        r"(?is)^(`[^`]+`|[A-Za-z_]\w*)\s*=\s*(\(.*)$", text
    )
    if plain_path:
        return (
            f"PATH {unquote_identifier(plain_path.group(1))} = "
            f"{convert_pattern_chain(plain_path.group(2))}"
        )
    return convert_pattern_chain(text)


def convert_pattern_chain(text: str) -> str:
    output: List[str] = []
    node_count = 0
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        if text[index] == "(":
            inner, index = extract_balanced(text, index, "(", ")")
            output.append(convert_node(inner[1:-1]))
            node_count += 1
            continue
        if text.startswith("<-[", index):
            end = find_closing(text, index + 2, "[", "]")
            if end + 1 >= len(text) or text[end + 1] != "-":
                raise ValueError("malformed incoming relationship")
            output.append(f"<-{convert_relationship(text[index + 3:end])}-")
            index = end + 2
            continue
        if text.startswith("-[", index):
            end = find_closing(text, index + 1, "[", "]")
            suffix = "->" if text.startswith("->", end + 1) else "-"
            if suffix == "-" and (end + 1 >= len(text) or text[end + 1] != "-"):
                raise ValueError("malformed relationship")
            output.append(f"-{convert_relationship(text[index + 2:end])}{suffix}")
            index = end + 3 if suffix == "->" else end + 2
            continue
        if text.startswith("<--", index):
            output.append("<--")
            index += 3
            continue
        if text.startswith("-->", index):
            output.append("-->")
            index += 3
            continue
        if text.startswith("--", index):
            output.append("--")
            index += 2
            continue
        raise ValueError(f"unexpected token near {text[index:index + 20]!r}")
    if node_count == 0:
        raise ValueError("empty or invalid graph pattern")
    return " ".join(output)


def convert_node(inner: str) -> str:
    map_text, prefix = extract_property_map(inner)
    prefix = prefix.strip()
    variable = ""
    labels: List[str] = []
    if prefix:
        parts = split_identifier_colons(prefix)
        if not prefix.startswith(":"):
            variable = unquote_identifier(parts.pop(0)) if parts else ""
        labels = [unquote_identifier(item) for item in parts if item]
    label_text = "".join(
        [f":{labels[0]}", *[f"&{label}" for label in labels[1:]]]
    ) if labels else ""
    properties = convert_property_map(map_text) if map_text is not None else ""
    if variable:
        return f"{variable}{label_text}{properties}"
    return f"({label_text}{properties})"


def convert_relationship(inner: str) -> str:
    map_text, prefix = extract_property_map(inner)
    hop_match = re.search(r"\*\s*(?:\d+)?(?:\.\.(?:\d+)?)?", prefix)
    hop = re.sub(r"\s+", "", hop_match.group(0)) if hop_match else ""
    if hop_match:
        prefix = prefix[: hop_match.start()] + prefix[hop_match.end() :]
    prefix = prefix.strip()
    variable = ""
    rel_types: List[str] = []
    if ":" in prefix:
        variable_part, type_part = prefix.split(":", 1)
        variable = unquote_identifier(variable_part.strip())
        rel_types = [
            unquote_identifier(item.strip())
            for item in type_part.split("|")
            if item.strip()
        ]
    elif prefix:
        variable = unquote_identifier(prefix)
    type_text = f":{'|'.join(rel_types)}" if rel_types else ""
    properties = convert_property_map(map_text) if map_text is not None else ""
    return f"{variable}{type_text}{hop}{properties}"


def convert_property_map(body: str) -> str:
    items = []
    for item in split_top_level(body):
        colon = find_top_level_colon(item)
        if colon < 0:
            raise ValueError(f"invalid property map item:{item}")
        key = unquote_identifier(item[:colon].strip())
        value = canonical_expression(item[colon + 1 :])
        items.append(f"{key} = {value}")
    return " {" + ", ".join(items) + "}"


def canonical_expression(text: str) -> str:
    value = canonicalize_strings(str(text).strip())
    value = re.sub(r"(?i)\bAND\b", "AND", value)
    value = re.sub(r"(?i)\bOR\b", "OR", value)
    value = re.sub(r"(?i)\bNOT\b", "NOT", value)
    value = re.sub(r"(?i)\bIS\s+NULL\b", "IS NULL", value)
    value = re.sub(r"(?i)\bIS\s+NOT\s+NULL\b", "IS NOT NULL", value)
    value = re.sub(r"(?i)\bSTARTS\s+WITH\b", "STARTS WITH", value)
    value = re.sub(r"(?i)\bENDS\s+WITH\b", "ENDS WITH", value)
    value = re.sub(r"(?i)\bCONTAINS\b", "CONTAINS", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def canonical_order_item(item: str) -> str:
    match = re.match(r"(?is)^(.+?)(?:\s+(ASC|DESC))?\s*$", item)
    if not match:
        return canonical_expression(item)
    return f"{canonical_expression(match.group(1))} {(match.group(2) or 'ASC').upper()}"


def parse_aggregate(text: str, default_alias: str = "") -> Optional[_Aggregate]:
    match = AGGREGATE_RE.match(text)
    if not match:
        return None
    function = match.group(1).upper()
    distinct = bool(match.group(2))
    argument = match.group(3).strip()
    alias = unquote_identifier(match.group(4) or default_alias)
    if argument == "*" and (function != "COUNT" or distinct):
        return None
    return _Aggregate(function, argument, distinct, alias)


def split_clauses(cypher: str) -> List[Tuple[str, str]]:
    starts: List[Tuple[int, int, str]] = []
    depths = {"(": 0, "[": 0, "{": 0}
    quote: Optional[str] = None
    index = 0
    while index < len(cypher):
        char = cypher[index]
        if quote:
            if char == quote and not is_escaped(cypher, index):
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char in "([{":
            depths[char] += 1
            index += 1
            continue
        if char in ")]}":
            opener = {")": "(", "]": "[", "}": "{"}[char]
            depths[opener] = max(0, depths[opener] - 1)
            index += 1
            continue
        if not any(depths.values()):
            keyword = match_keyword(cypher, index)
            if keyword:
                if keyword == "WITH":
                    previous = re.search(r"([A-Za-z]+)\s*$", cypher[:index])
                    if previous and previous.group(1).upper() in {"STARTS", "ENDS"}:
                        index += 1
                        continue
                starts.append((index, index + len(keyword), keyword))
                index += len(keyword)
                continue
        index += 1
    clauses = []
    for offset, (start, body_start, keyword) in enumerate(starts):
        end = starts[offset + 1][0] if offset + 1 < len(starts) else len(cypher)
        clauses.append((keyword, cypher[body_start:end].strip()))
    return clauses


def match_keyword(text: str, index: int) -> Optional[str]:
    if index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_"):
        return None
    for keyword in CLAUSE_KEYWORDS:
        end = index + len(keyword)
        if text[index:end].upper() != keyword:
            continue
        if end < len(text) and (text[end].isalnum() or text[end] == "_"):
            continue
        return keyword
    return None


def split_union_branches(text: str) -> Tuple[List[str], bool]:
    positions: List[Tuple[int, int, bool]] = []
    depths = {"(": 0, "[": 0, "{": 0}
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote and not is_escaped(text, index):
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char in "([{":
            depths[char] += 1
        elif char in ")]}":
            opener = {")": "(", "]": "[", "}": "{"}[char]
            depths[opener] = max(0, depths[opener] - 1)
        elif not any(depths.values()):
            keyword = match_keyword(text, index)
            if keyword in {"UNION", "UNION ALL"}:
                positions.append((index, index + len(keyword), keyword == "UNION ALL"))
                index += len(keyword)
                continue
        index += 1
    if not positions:
        return [text.strip()], False
    branches = []
    start = 0
    union_all = True
    for position, end, is_all in positions:
        branches.append(text[start:position].strip())
        start = end
        union_all = union_all and is_all
    branches.append(text[start:].strip())
    return branches, union_all


def split_top_level(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    quote: Optional[str] = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and not is_escaped(text, index):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char in "([{":
            depths[char] += 1
        elif char in ")]}":
            opener = {")": "(", "]": "[", "}": "{"}[char]
            depths[opener] = max(0, depths[opener] - 1)
        elif char == "," and not any(depths.values()):
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def extract_property_map(text: str) -> Tuple[Optional[str], str]:
    quote: Optional[str] = None
    start: Optional[int] = None
    depth = 0
    for index, char in enumerate(text):
        if quote:
            if char == quote and not is_escaped(text, index):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "{":
            if start is None:
                start = index
            depth += 1
        elif char == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], (text[:start] + text[index + 1 :])
    return None, text


def extract_balanced(
    text: str, start: int, opener: str, closer: str
) -> Tuple[str, int]:
    end = find_closing(text, start, opener, closer)
    return text[start : end + 1], end + 1


def find_closing(text: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    quote: Optional[str] = None
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if char == quote and not is_escaped(text, index):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed {opener}")


def find_top_level_colon(text: str) -> int:
    depth = 0
    quote: Optional[str] = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and not is_escaped(text, index):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return index
    return -1


def split_identifier_colons(text: str) -> List[str]:
    return [item.strip() for item in re.split(r"\s*:\s*", text)]


def canonicalize_strings(text: str) -> str:
    output: List[str] = []
    index = 0
    while index < len(text):
        if text[index] not in {"'", '"'}:
            output.append(text[index])
            index += 1
            continue
        quote = text[index]
        index += 1
        value: List[str] = []
        while index < len(text):
            char = text[index]
            if char == quote and not is_escaped(text, index):
                index += 1
                break
            if char == "\\" and index + 1 < len(text):
                index += 1
                value.append(text[index])
                index += 1
                continue
            value.append(char)
            index += 1
        output.append(json.dumps("".join(value), ensure_ascii=False))
    return "".join(output)


def parse_nonnegative_integer(
    text: str, clause: str, result: ConversionResult
) -> Optional[int]:
    match = re.fullmatch(r"\s*(\d+)\s*", text)
    if not match:
        result.reject(f"non_literal_{clause.lower()}")
        return None
    return int(match.group(1))


def is_escaped(text: str, index: int) -> bool:
    count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        count += 1
        index -= 1
    return count % 2 == 1


def unquote_identifier(value: str) -> str:
    return str(value or "").strip().strip("`")


def clean_cypher(cypher: str) -> str:
    text = str(cypher or "").strip().rstrip(";")
    text = re.sub(r"^```(?:cypher)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def convert_cypher_to_sirc(cypher: str) -> ConversionResult:
    return SIRCConverter().convert(cypher)


convert_cypher_to_natcypher = convert_cypher_to_sirc
NatCypherConverter = SIRCConverter
