from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from torch_lattice_conformance import e2e, generate, migration

_Command = tuple[str, str, Callable[[], None]]
_COMMANDS: tuple[_Command, ...] = (
    ('fuzz', 'Generate randomized CUDA provenance archives for MLX replay.', generate.main),
    ('e2e-fixtures', 'Write fixed CUDA-to-MLX regression fixtures.', e2e.main),
    ('migration', 'Compare the supported original TorchSparse migration subset.', migration.main),
)


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {'-h', '--help'}:
        _print_help()
        return

    command = args.pop(0)
    for name, _help, entry in _COMMANDS:
        if command == name:
            sys.argv = [f'conformance {name}', *args]
            entry()
            return

    choices = ', '.join(name for name, _help, _entry in _COMMANDS)
    raise SystemExit(f'unknown conformance command {command!r}; choose one of: {choices}')


def _print_help() -> None:
    parser = argparse.ArgumentParser(
        prog='conformance',
        description='Torch-side lattice artifact conformance tools.',
    )
    subcommands = parser.add_subparsers(dest='command')
    for name, help_text, _entry in _COMMANDS:
        subcommands.add_parser(name, help=help_text)
    parser.print_help()


if __name__ == '__main__':
    main()
