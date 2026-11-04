#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Cypher to SIRC, or compile SIRC to Cypher.")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert a Cypher query to SIRC")
    convert.add_argument("cypher", nargs="?", help="Cypher text. Reads stdin if omitted.")
    convert.add_argument("--json", action="store_true", help="Print the full conversion record.")

    compile_cmd = sub.add_parser("compile", help="Compile SIRC to Cypher")
    compile_cmd.add_argument("sirc", nargs="?", help="SIRC text. Reads stdin if omitted.")
    compile_cmd.add_argument("--schema", default=None, help="Optional compact schema text.")
    compile_cmd.add_argument("--no-validate", action="store_true", help="Skip schema validation.")
    compile_cmd.add_argument("--json", action="store_true", help="Print the full compilation record.")
    return parser


def _read_text(value: str | None) -> str:
    if value:
        return value
    return sys.stdin.read()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        from sirc.generator import convert_cypher_to_sirc

        result = convert_cypher_to_sirc(_read_text(args.cypher))
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        elif result.supported:
            print(result.sirc)
        else:
            print("unsupported:", "; ".join(result.errors), file=sys.stderr)
            return 2
        return 0 if result.supported else 2

    from sirc.compiler import compile_sirc

    result = compile_sirc(
        _read_text(args.sirc),
        schema=args.schema,
        validate=not args.no_validate,
    )
    if args.json:
        payload = {
            "cypher": result.cypher,
            "supported": result.supported,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif result.supported:
        print(result.cypher)
    else:
        print("unsupported:", "; ".join(result.errors), file=sys.stderr)
        return 2
    return 0 if result.supported else 2


if __name__ == "__main__":
    sys.exit(main())
