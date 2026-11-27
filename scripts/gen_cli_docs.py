#!/usr/bin/env python3
"""Regenerate docs/cli.md from the argparse definitions: python3 scripts/gen_cli_docs.py > docs/cli.md"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from conjectra.cli import build_parser  # noqa: E402


def main() -> None:
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {ca.dest: ca.help for ca in sub._choices_actions}
    print("# CLI reference\n")
    print("Generated from `conjectra --help`. Every command accepts `--home DIR` (alias `--arr-home`).")
    print("Commands print JSON unless noted.\n")
    print("| Command | Purpose |\n|---|---|")
    for name in sub.choices:
        print(f"| [`{name}`](#{name}) | {helps.get(name, '')} |")
    print()
    for name, sp in sub.choices.items():
        print(f"## {name}\n")
        if helps.get(name):
            print(helps[name] + "\n")
        print("```")
        print(sp.format_help().replace(sp.prog, f"conjectra {name}", 1).rstrip())
        print("```\n")


if __name__ == "__main__":
    main()
