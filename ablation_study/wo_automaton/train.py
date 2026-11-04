#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (
    DEFAULT_STAGE1_DIR,
    DEFAULT_STAGE2_DIR,
    add_lora_train_arguments,
    lora_train_argv,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2"), required=True)
    add_lora_train_arguments(parser, output_dir=DEFAULT_STAGE1_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.stage == "1":
        from natcypher.train_ir_lora import main as train_stage1

        output_dir = DEFAULT_STAGE1_DIR if args.output_dir == DEFAULT_STAGE1_DIR else args.output_dir
        train_stage1(lora_train_argv(args, output_dir=output_dir, flavor="ir"))
        return

    from independent_two_lora.train_stage2_lora import main as train_stage2

    output_dir = (
        DEFAULT_STAGE2_DIR if args.output_dir == DEFAULT_STAGE1_DIR else args.output_dir
    )
    train_stage2(lora_train_argv(args, output_dir=output_dir, flavor="stage2"))


if __name__ == "__main__":
    main()
