"""Branded command-line alias for the SGLang-compatible runtime."""

from sglang.cli.main import main as _sglang_main


def main():
    _sglang_main(prog="langslanger")
