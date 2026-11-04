# SIRC

SIRC (Structured Intermediate Representation for Cypher) is the query IR used by SAGCypher.

This package converts gold Cypher to SIRC, parses and validates SIRC.

## Layout

| File | Role |
|:--|:--|
| `generator.py` | Cypher → SIRC |
| `parser.py` | SIRC text → AST |
| `validator.py` | scope / schema checks |
| `compiler.py` | SIRC AST → Cypher |
| `serializer.py` | canonical SIRC spelling |
| `ast.py` | typed syntax tree |
| `generate_training_data.py` | convert a dataset's gold Cypher to SIRC JSONL |
| `coverage.py` | batch coverage / compile audit |
| `audit.py` | check that stored SIRC is canonical |

## Python API

```python
from sirc import convert_cypher_to_sirc, compile_sirc

cypher = "MATCH (n:Movie) WHERE n.year > 2000 RETURN n.title"

converted = convert_cypher_to_sirc(cypher)
if not converted.supported:
    raise ValueError(converted.errors)
print(converted.sirc)

compiled = compile_sirc(converted.sirc)
if not compiled.supported:
    raise ValueError(compiled.errors)
print(compiled.cypher)
```

`convert_cypher_to_sirc()` never treats a partial conversion as a training target. If a construct is outside the supported read-only subset, `supported` is `False` and `errors` explains why.

## Command line

From the repository root:

```bash
# Cypher → SIRC
python -m sirc convert 'MATCH (n:Movie) RETURN n.title'

# Convert a dataset (writes sirc_data/<dataset>.jsonl)
python -m sirc.generate_training_data --dataset cypherbench --limit 20
python -m sirc.generate_training_data --dataset all
```

## Supported Cypher subset

SIRC targets read-only Cypher queries: `MATCH` / `OPTIONAL MATCH`, `WHERE`, `WITH`, `UNWIND`, `CALL` , `RETURN`, `ORDER BY`, `SKIP` / `LIMIT`, and `UNION`. Write clauses such as `CREATE`, `MERGE`, `SET`, and `DELETE` are rejected.
