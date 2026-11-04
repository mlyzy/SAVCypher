from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class UnifiedText2CypherExample:
    example_id: str
    question: str
    gold_cypher: str
    graph_key: str
    schema_text: str = ""
    schema_struct: Dict[str, Any] = field(default_factory=dict)
    task_type: str = ""
    prediction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
