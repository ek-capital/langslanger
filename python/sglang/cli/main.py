import argparse
import sys
from pathlib import Path

from sglang.cli.utils import get_git_commit_hash
from sglang.version import __version__


def version(args, extra_argv):
    program_name = getattr(args, "program_name", "sglang")
    print(f"{program_name} version: {__version__}")
    if program_name == "langslanger":
        print("sglang compatibility: enabled")
    print(f"git revision: {get_git_commit_hash()[:7]}")


def main(prog=None):
    invoked_name = Path(sys.argv[0]).name
    is_langslanger = prog == "langslanger" or invoked_name == "langslanger"
    program_name = "langslanger" if is_langslanger else "sglang"
    parser = argparse.ArgumentParser(prog=program_name)

    # complex sub commands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser(
        "serve",
        help="Launch an SGLang server.",
        add_help=False,
    )
    subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,
    )

    # simple commands
    version_parser = subparsers.add_parser(
        "version",
        help="Show the version information.",
    )
    version_parser.set_defaults(func=version, program_name=program_name)

    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "serve":
        from sglang.cli.serve import serve

        serve(args, extra_argv)
    elif args.subcommand == "generate":
        from sglang.cli.generate import generate

        generate(args, extra_argv)
    elif args.subcommand == "version":
        version(args, extra_argv)
