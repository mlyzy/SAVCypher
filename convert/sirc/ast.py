from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class Expression:
    text: str


@dataclass(frozen=True)
class ResultItem:
    expression: Expression
    alias: Optional[str] = None


@dataclass(frozen=True)
class Node:
    variable: Optional[str]
    labels: tuple[str, ...] = ()
    properties: tuple[tuple[str, Expression], ...] = ()
    anonymous: bool = False


@dataclass(frozen=True)
class Relationship:
    variable: Optional[str]
    types: tuple[str, ...]
    direction: str
    hop: Optional[str] = None
    properties: tuple[tuple[str, Expression], ...] = ()


@dataclass(frozen=True)
class Pattern:
    nodes: tuple[Node, ...]
    relationships: tuple[Relationship, ...]
    path_variable: Optional[str] = None
    path_mode: Optional[str] = None


@dataclass(frozen=True)
class Answer:
    items: tuple[ResultItem, ...]
    distinct: bool = False


@dataclass(frozen=True)
class Graph:
    patterns: tuple[Pattern, ...]


@dataclass(frozen=True)
class Filter:
    predicate: Expression


@dataclass(frozen=True)
class Let:
    bindings: tuple[ResultItem, ...]


@dataclass(frozen=True)
class Keep:
    items: tuple[ResultItem, ...]


@dataclass(frozen=True)
class AggregateBinding:
    call: Expression
    alias: str


@dataclass(frozen=True)
class GroupKey:
    expression: Expression
    alias: Optional[str] = None


@dataclass(frozen=True)
class Aggregate:
    bindings: tuple[AggregateBinding, ...]
    per: tuple[GroupKey, ...] = ()
    having: Optional[Expression] = None


@dataclass(frozen=True)
class Dedup:
    keys: tuple[Expression, ...]


@dataclass(frozen=True)
class Expand:
    expression: Expression
    variable: str


@dataclass(frozen=True)
class OptionalBlock:
    steps: tuple["Step", ...]


@dataclass(frozen=True)
class Scope:
    steps: tuple["Step", ...]
    expose: tuple[str, ...]


@dataclass(frozen=True)
class AnyBranch:
    steps: tuple["Step", ...]


@dataclass(frozen=True)
class AnyClause:
    branches: tuple[AnyBranch, ...]
    expose: tuple[str, ...]
    all: bool = False


@dataclass(frozen=True)
class OrderItem:
    expression: Expression
    direction: str = "ASC"


@dataclass(frozen=True)
class Order:
    items: tuple[OrderItem, ...]


@dataclass(frozen=True)
class Top:
    limit: int
    items: tuple[OrderItem, ...]
    offset: Optional[int] = None


@dataclass(frozen=True)
class Take:
    limit: int


@dataclass(frozen=True)
class Offset:
    offset: int


Step = Union[
    Graph,
    Filter,
    Let,
    Keep,
    Aggregate,
    Dedup,
    Expand,
    OptionalBlock,
    Scope,
    AnyClause,
    Order,
    Top,
    Take,
    Offset,
]


@dataclass(frozen=True)
class Query:
    answer: Answer
    steps: tuple[Step, ...] = field(default_factory=tuple)
