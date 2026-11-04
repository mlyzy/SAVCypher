from __future__ import annotations

import re
from typing import Iterable, Optional

from sirc.ast import (
    Aggregate,
    AggregateBinding,
    Answer,
    AnyBranch,
    AnyClause,
    Dedup,
    Expand,
    Expression,
    Filter,
    Graph,
    GroupKey,
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
    ResultItem,
    Scope,
    Take,
    Top,
)


IDENT = r"(?:`[^`]+`|[A-Za-z_]\w*)"
CLAUSE_RE = re.compile(
    r"^(ANSWER|GRAPH|FILTER|LET|KEEP|AGG|HAVING|DEDUP\s+BY|EXPAND|"
    r"OPTIONAL|ANY(?:\s+ALL)?|SCOPE|EXPOSE|ORDER\s+BY|TOP|TAKE|"
    r"LIMIT|SKIP|OFFSET)\b",
    re.I,
)


class SIRCSyntaxError(ValueError):
    """A deterministic parse failure with source context."""


def _ident(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(IDENT, value):
        raise SIRCSyntaxError(f"invalid identifier: {value!r}")
    return value[1:-1] if value.startswith("`") else value


def split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    quote: Optional[str] = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char in "([{":
            depths[char] += 1
        elif char in ")]}":
            opener = {")": "(", "]": "[", "}": "{"}[char]
            depths[opener] -= 1
            if depths[opener] < 0:
                raise SIRCSyntaxError(f"unmatched {char!r}")
        elif char == separator and not any(depths.values()):
            parts.append(text[start:index].strip())
            start = index + 1
    if quote or any(depths.values()):
        raise SIRCSyntaxError("unclosed quote or delimiter")
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_alias(text: str) -> tuple[str, Optional[str]]:
    text = _strip_line_comment(text).strip()
    matches = list(re.finditer(rf"(?i)\s+AS\s+({IDENT})\s*$", text))
    if not matches:
        return text.strip(), None
    match = matches[-1]
    return text[: match.start()].strip(), _ident(match.group(1))


def _strip_line_comment(text: str) -> str:
    quote: Optional[str] = None
    index = 0
    while index < len(text) - 1:
        char = text[index]
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
        elif char in {"'", '"', "`"}:
            quote = char
        elif text.startswith("//", index):
            return text[:index]
        index += 1
    return text


def _items(text: str) -> tuple[ResultItem, ...]:
    result = []
    for item in split_top_level(text):
        expression, alias = _split_alias(item)
        if not expression:
            raise SIRCSyntaxError("empty expression")
        result.append(ResultItem(Expression(expression), alias))
    if not result:
        raise SIRCSyntaxError("empty item list")
    return tuple(result)


def _property_map(text: str) -> tuple[str, tuple[tuple[str, Expression], ...]]:
    text = text.strip()
    if not text.endswith("}"):
        return text, ()
    depth = 0
    quote: Optional[str] = None
    start = None
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                start = index
                break
    if start is None:
        raise SIRCSyntaxError("unclosed property map")
    properties = []
    for item in split_top_level(text[start + 1 : -1]):
        if "=" not in item:
            raise SIRCSyntaxError(f"invalid property map item: {item!r}")
        key, value = item.split("=", 1)
        properties.append((_ident(key), Expression(value.strip())))
    return text[:start].strip(), tuple(properties)


def _parse_node(text: str, anonymous: bool = False) -> Node:
    prefix, properties = _property_map(text)
    parts = re.split(r"\s*(?::|&)\s*", prefix)
    variable: Optional[str] = None
    labels: list[str] = []
    if prefix.startswith(":"):
        labels = [_ident(item) for item in parts if item]
    elif prefix:
        variable = _ident(parts[0])
        labels = [_ident(item) for item in parts[1:] if item]
    if not variable and not labels and not anonymous:
        raise SIRCSyntaxError("named node is empty")
    return Node(variable, tuple(labels), properties, anonymous)


def _parse_relationship(text: str, direction: str) -> Relationship:
    prefix, properties = _property_map(text)
    hop = None
    hop_match = re.search(r"\*(?:\d+)?(?:\.\.(?:\d+)?)?", prefix)
    if hop_match:
        hop = hop_match.group(0)
        prefix = prefix[: hop_match.start()] + prefix[hop_match.end() :]
    variable = None
    types: list[str] = []
    if ":" in prefix:
        variable_text, type_text = prefix.split(":", 1)
        variable = _ident(variable_text) if variable_text.strip() else None
        types = [_ident(item.lstrip(":")) for item in type_text.split("|") if item.strip()]
    elif prefix.strip():
        variable = _ident(prefix)
    return Relationship(variable, tuple(types), direction, hop, properties)


def parse_pattern(text: str) -> Pattern:
    text = text.strip()
    path_variable = path_mode = None
    path = re.match(
        rf"(?is)^PATH\s+({IDENT})\s*=\s*(?:(SHORTEST|ALL_SHORTEST)\s+)?(.+)$",
        text,
    )
    if path:
        path_variable, path_mode, text = _ident(path.group(1)), path.group(2), path.group(3)
    nodes: list[Node] = []
    relationships: list[Relationship] = []
    index = 0

    def node_at(position: int) -> tuple[Node, int]:
        if text[position] == "(":
            end = _closing(text, position, "(", ")")
            return _parse_node(text[position + 1 : end], anonymous=True), end + 1
        match = re.match(
            rf"{IDENT}(?:(?::|&){IDENT})*(?:\s*\{{(?:[^{{}}]|\"[^\"]*\"|'[^']*')*\}})?",
            text[position:],
        )
        if not match:
            raise SIRCSyntaxError(f"expected node near {text[position:position+30]!r}")
        return _parse_node(match.group(0)), position + len(match.group(0))

    while index < len(text) and text[index].isspace():
        index += 1
    node, index = node_at(index)
    nodes.append(node)
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        if text.startswith("<-", index):
            direction, body_start = "in", index + 2
            end = text.find("-", body_start)
            if end < 0:
                raise SIRCSyntaxError("malformed incoming relationship")
            body, index = text[body_start:end], end + 1
        elif text.startswith("-", index):
            body_start = index + 1
            arrow = _relationship_end(text, body_start)
            body, direction, index = text[body_start:arrow[0]], arrow[1], arrow[2]
        else:
            raise SIRCSyntaxError(f"expected relationship near {text[index:index+30]!r}")
        relationships.append(_parse_relationship(body.strip(), direction))
        while index < len(text) and text[index].isspace():
            index += 1
        node, index = node_at(index)
        nodes.append(node)
    if len(nodes) != len(relationships) + 1:
        raise SIRCSyntaxError("incomplete pattern")
    return Pattern(tuple(nodes), tuple(relationships), path_variable, path_mode)


def _relationship_end(text: str, start: int) -> tuple[int, str, int]:
    depth = 0
    quote: Optional[str] = None
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if char == quote and text[index - 1] != "\\":
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and text.startswith("->", index):
            return index, "out", index + 2
        elif depth == 0 and char == "-":
            return index, "undirected", index + 1
    raise SIRCSyntaxError("malformed relationship")


def _closing(text: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    quote: Optional[str] = None
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if char == quote and text[index - 1] != "\\":
                quote = None
        elif char in {"'", '"', "`"}:
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    raise SIRCSyntaxError(f"unclosed {opener}")


def _orders(text: str) -> tuple[OrderItem, ...]:
    result = []
    for item in split_top_level(text):
        match = re.match(r"(?is)^(.+?)(?:\s+(ASC|DESC))?$", item)
        if not match:
            raise SIRCSyntaxError(f"invalid order item: {item!r}")
        result.append(OrderItem(Expression(match.group(1).strip()), (match.group(2) or "ASC").upper()))
    return tuple(result)


def _consume_block(lines: list[str], index: int) -> tuple[list[str], str, int]:
    joined = "\n".join(lines[index:])
    opening = joined.find("{")
    if opening < 0:
        raise SIRCSyntaxError(f"expected block after {lines[index]!r}")
    closing = _closing(joined, opening, "{", "}")
    prefix = joined[:opening].strip()
    body = joined[opening + 1 : closing]
    consumed_chars = closing + 1
    consumed_lines = joined[:consumed_chars].count("\n") + 1
    suffix_line = joined[consumed_chars:].split("\n", 1)[0].strip()
    return body.splitlines(), (prefix + " " + suffix_line).strip(), index + consumed_lines


def _continuation(lines: list[str], index: int) -> tuple[str, int]:
    values = [lines[index].strip()]
    index += 1
    while index < len(lines) and lines[index].strip() and not CLAUSE_RE.match(lines[index].strip()):
        values.append(lines[index].strip())
        index += 1
    return " ".join(values), index


def _parse_steps(lines: Iterable[str]) -> tuple:
    lines = [line.rstrip() for line in lines if line.strip()]
    steps = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        upper = line.upper()
        if upper.startswith("GRAPH"):
            first = line[5:].strip()
            chunks = [first] if first else []
            index += 1
            while index < len(lines) and not CLAUSE_RE.match(lines[index].strip()):
                chunks.append(lines[index].strip())
                index += 1
            patterns = []
            for chunk in chunks:
                patterns.extend(parse_pattern(item) for item in split_top_level(chunk, ";") if item)
            if not patterns:
                raise SIRCSyntaxError("GRAPH requires a pattern")
            steps.append(Graph(tuple(patterns)))
        elif upper.startswith("FILTER"):
            text, index = _continuation(lines, index)
            steps.append(Filter(Expression(text[6:].strip())))
        elif upper.startswith("LET "):
            steps.append(Let(_items(line[4:])))
            index += 1
        elif upper.startswith("KEEP "):
            steps.append(Keep(_items(line[5:])))
            index += 1
        elif upper.startswith("DEDUP BY "):
            steps.append(Dedup(tuple(Expression(item) for item in split_top_level(line[9:]))))
            index += 1
        elif upper.startswith("EXPAND "):
            expression, alias = _split_alias(line[7:])
            if not alias:
                raise SIRCSyntaxError("EXPAND requires AS variable")
            steps.append(Expand(Expression(expression), alias))
            index += 1
        elif upper.startswith("AGG "):
            text = line[4:].strip()
            index += 1
            having = None
            if index < len(lines) and lines[index].strip().upper().startswith("HAVING "):
                having = Expression(lines[index].strip()[7:].strip())
                index += 1
            per_parts = re.split(r"(?i)\s+PER\s+", text, maxsplit=1)
            bindings = []
            for item in split_top_level(per_parts[0]):
                call, alias = _split_alias(item)
                if not alias or not re.match(r"(?i)^(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(", call):
                    raise SIRCSyntaxError(f"invalid aggregate binding: {item!r}")
                bindings.append(AggregateBinding(Expression(call), alias))
            per = (
                tuple(
                    GroupKey(item.expression, item.alias)
                    for item in _items(per_parts[1])
                )
                if len(per_parts) == 2
                else ()
            )
            steps.append(Aggregate(tuple(bindings), per, having))
        elif upper.startswith("OPTIONAL"):
            body, _, index = _consume_block(lines, index)
            steps.append(OptionalBlock(_parse_pattern_block(body)))
        elif upper.startswith("SCOPE"):
            body, trailer, index = _consume_block(lines, index)
            expose = re.search(r"(?i)\bEXPOSE\s+(.+)$", trailer)
            if not expose:
                raise SIRCSyntaxError("SCOPE requires EXPOSE")
            steps.append(Scope(_parse_steps(body), tuple(_ident(v) for v in split_top_level(expose.group(1)))))
        elif upper.startswith("ANY"):
            all_rows = upper.startswith("ANY ALL")
            branches = []
            while True:
                body, trailer, index = _consume_block(lines, index)
                branches.append(AnyBranch(_parse_pattern_block(body)))
                expose = re.search(r"(?i)\bEXPOSE\s+(.+)$", trailer)
                if not expose and index < len(lines) and lines[index].strip().upper().startswith("EXPOSE "):
                    expose = re.search(r"(?i)^EXPOSE\s+(.+)$", lines[index].strip())
                    index += 1
                if expose:
                    exposed = tuple(_ident(v) for v in split_top_level(expose.group(1)))
                    break
                if index >= len(lines) or not lines[index].strip().upper().startswith("OR"):
                    raise SIRCSyntaxError("ANY requires OR branches and EXPOSE")
                # The next block may start on the OR line or the following line.
                if "{" not in lines[index]:
                    index += 1
            steps.append(AnyClause(tuple(branches), exposed, all_rows))
        elif upper.startswith("ORDER BY "):
            steps.append(Order(_orders(line[9:])))
            index += 1
        elif upper.startswith("TOP "):
            match = re.match(r"(?is)^TOP\s+(\d+)\s+BY\s+(.+?)(?:\s+OFFSET\s+(\d+))?$", line)
            if not match:
                raise SIRCSyntaxError(f"invalid TOP clause: {line!r}")
            steps.append(Top(int(match.group(1)), _orders(match.group(2)), int(match.group(3)) if match.group(3) else None))
            index += 1
        elif upper.startswith(("TAKE ", "LIMIT ")):
            steps.append(Take(int(line.split(None, 1)[1])))
            index += 1
        elif upper.startswith(("SKIP ", "OFFSET ")):
            steps.append(Offset(int(line.split(None, 1)[1])))
            index += 1
        else:
            raise SIRCSyntaxError(f"unsupported or misplaced clause: {line!r}")
    return tuple(steps)


def _parse_pattern_block(lines: Iterable[str]) -> tuple:
    """Parse a block whose leading patterns omit the GRAPH keyword."""
    values = [line.rstrip() for line in lines if line.strip()]
    if values and not CLAUSE_RE.match(values[0].strip()):
        values[0] = "GRAPH " + values[0].strip()
    return _parse_steps(values)


def parse_sirc(text: str) -> Query:
    """Parse canonical SIRC into a typed syntax tree."""
    lines = [line.rstrip() for line in str(text or "").strip().splitlines() if line.strip()]
    if not lines or not lines[0].strip().upper().startswith("ANSWER "):
        raise SIRCSyntaxError("query must start with ANSWER")
    answer_text = lines[0].strip()[7:].strip()
    distinct = bool(re.match(r"(?i)^DISTINCT\b", answer_text))
    if distinct:
        answer_text = re.sub(r"(?i)^DISTINCT\s+", "", answer_text, count=1)
    return Query(Answer(_items(answer_text), distinct), _parse_steps(lines[1:]))


parse_natcypher = parse_sirc
NatCypherSyntaxError = SIRCSyntaxError
