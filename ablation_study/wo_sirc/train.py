#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import DEFAULT_WO_SIRC_DIR, add_lora_train_arguments, lora_train_argv
from natcypher.train_direct_lora import main as train_direct


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_lora_train_arguments(parser, output_dir=DEFAULT_WO_SIRC_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    train_direct(lora_train_argv(args, flavor="direct"))


if __name__ == "__main__":
    main()
