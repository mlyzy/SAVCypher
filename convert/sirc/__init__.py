from sirc.ast import Query
from sirc.compiler import (
    CompilationResult,
    NatCypherCompileError,
    SIRCCompileError,
    compile_natcypher,
    compile_query,
    compile_sirc,
)
from sirc.generator import (
    ConversionResult,
    NatCypherConverter,
    SIRCConverter,
    convert_cypher_to_natcypher,
    convert_cypher_to_sirc,
)
from sirc.parser import (
    NatCypherSyntaxError,
    SIRCSyntaxError,
    parse_natcypher,
    parse_sirc,
)
from sirc.serializer import canonicalize_natcypher, canonicalize_sirc, serialize_query
from sirc.validator import ValidationResult, parse_compact_schema, validate_query

__all__ = [
    "CompilationResult",
    "ConversionResult",
    "NatCypherCompileError",
    "NatCypherConverter",
    "NatCypherSyntaxError",
    "Query",
    "SIRCCompileError",
    "SIRCConverter",
    "SIRCSyntaxError",
    "ValidationResult",
    "canonicalize_natcypher",
    "canonicalize_sirc",
    "compile_natcypher",
    "compile_query",
    "compile_sirc",
    "convert_cypher_to_natcypher",
    "convert_cypher_to_sirc",
    "parse_compact_schema",
    "parse_natcypher",
    "parse_sirc",
    "serialize_query",
    "validate_query",
]


def ir_text(record: dict) -> str:
    """Read SIRC text from a record, accepting the older NatCypher field names."""
    return str(
        record.get("sirc")
        or record.get("natcypher")
        or record.get("predicted_sirc")
        or record.get("predicted_natcypher")
        or ""
    )
