"""Compatibility wrapper for ``python -m langslanger.launch_server``."""

import runpy


def main():
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
