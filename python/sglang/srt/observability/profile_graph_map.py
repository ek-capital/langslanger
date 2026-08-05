"""Map Nsight Systems CUDA graph nodes to observed kernels and declared sources.

This is a supplemental, offline attribution tool. It does not contribute timing
to the authoritative serving profile report. Export an Nsight Systems report to
SQLite first, then optionally pass a LangSlanger profile report to join kernel
symbols to explicit implementation/source declarations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

GRAPH_MAP_SCHEMA_VERSION = 1
_KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
_NODE_TABLE = "CUDA_GRAPH_NODE_EVENTS"
_STRING_TABLE = "StringIds"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def analyze_graph_sqlite(
    sqlite_path: str | Path, profile_report_path: str | Path | None = None
) -> dict[str, Any]:
    """Return graph-node/kernel/source attribution from an Nsight SQLite export."""
    sqlite_path = Path(sqlite_path).expanduser().resolve()
    if not sqlite_path.is_file():
        raise ValueError(f"Nsight SQLite export does not exist: {sqlite_path}")

    report, report_path = _load_profile_report(profile_report_path)
    with _connect_read_only(sqlite_path) as connection:
        tables = _table_names(connection)
        if _KERNEL_TABLE not in tables:
            raise ValueError(
                f"{sqlite_path.name} has no {_KERNEL_TABLE} table; export a report "
                "collected with CUDA tracing enabled"
            )
        kernel_columns = _columns(connection, _KERNEL_TABLE)
        required = {"start", "end", "graphNodeId"}
        missing = required - kernel_columns
        if missing:
            raise ValueError(
                f"{_KERNEL_TABLE} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        strings = _string_ids(connection) if _STRING_TABLE in tables else {}
        original_nodes = _original_node_ids(connection) if _NODE_TABLE in tables else {}
        kernels = _read_graph_kernels(connection, kernel_columns, strings)
        export_schema_version = _export_schema_version(connection, tables)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    kernels_without_graph_node = 0
    for kernel in kernels:
        node_id = kernel["graph_node_id"]
        if node_id is None:
            kernels_without_graph_node += 1
            continue
        canonical_node_id = _canonical_node_id(node_id, original_nodes)
        key = (
            kernel.get("global_pid"),
            kernel.get("graph_id"),
            canonical_node_id,
        )
        groups[key].append(kernel)

    nodes = []
    for (global_pid, graph_id, canonical_node_id), values in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        symbols = defaultdict(lambda: {"occurrences": 0, "summed_kernel_ms": 0.0})
        observed_node_ids = set()
        devices = set()
        contexts = set()
        streams = set()
        launch_shapes = set()
        for kernel in values:
            symbol = kernel["symbol"]
            symbols[symbol]["occurrences"] += 1
            symbols[symbol]["summed_kernel_ms"] += kernel["duration_ns"] / 1e6
            observed_node_ids.add(kernel["graph_node_id"])
            _add_not_none(devices, kernel.get("device_id"))
            _add_not_none(contexts, kernel.get("context_id"))
            _add_not_none(streams, kernel.get("stream_id"))
            if kernel["launch_shape"]:
                launch_shapes.add(kernel["launch_shape"])

        symbol_rows = [
            {"symbol": symbol, **metrics}
            for symbol, metrics in sorted(
                symbols.items(),
                key=lambda item: item[1]["summed_kernel_ms"],
                reverse=True,
            )
        ]
        source_candidates = _source_candidates(symbol_rows, report)
        source_mapping_status = {
            0: "unresolved",
            1: "single_candidate",
        }.get(len(source_candidates), "ambiguous")
        nodes.append(
            {
                "global_pid": global_pid,
                "graph_id": graph_id,
                "canonical_graph_node_id": canonical_node_id,
                "observed_graph_node_ids": sorted(observed_node_ids),
                "kernel_occurrences": len(values),
                "summed_kernel_ms": sum(kernel["duration_ns"] for kernel in values)
                / 1e6,
                "symbols": symbol_rows,
                "device_ids": sorted(devices),
                "context_ids": sorted(contexts),
                "stream_ids": sorted(streams),
                "launch_shapes": sorted(launch_shapes),
                "source_mapping_status": source_mapping_status,
                "source_candidates": source_candidates,
            }
        )

    warnings = []
    if not groups:
        warnings.append(
            "no kernels carried graphNodeId; collect Nsight CUDA graph trace at "
            "node granularity"
        )
    if _NODE_TABLE not in tables:
        warnings.append(
            f"{_NODE_TABLE} is absent; cloned nodes cannot be folded to originals"
        )
    if report is None:
        warnings.append(
            "no LangSlanger profile report supplied; source candidates are unavailable"
        )

    return {
        "schema_version": GRAPH_MAP_SCHEMA_VERSION,
        "timing_authority": "supplemental_nsight_kernel_activity",
        "authoritative_serving_timing": False,
        "input": {
            "nsight_sqlite": str(sqlite_path),
            "nsight_sqlite_sha256": _sha256(sqlite_path),
            "nsight_export_schema_version": export_schema_version,
            "profile_report": str(report_path) if report_path else None,
            "profile_report_sha256": _sha256(report_path) if report_path else None,
            "profile_id": report.get("profile_id") if report else None,
        },
        "coverage": {
            "kernel_rows": len(kernels),
            "graph_kernel_rows": len(kernels) - kernels_without_graph_node,
            "kernels_without_graph_node": kernels_without_graph_node,
            "canonical_graph_nodes": len(nodes),
        },
        "graph_nodes": nodes,
        "warnings": warnings,
    }


def write_graph_map(
    sqlite_path: str | Path,
    *,
    profile_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    result = analyze_graph_sqlite(sqlite_path, profile_report_path)
    sqlite_path = Path(sqlite_path).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else sqlite_path.with_suffix(".graph-map.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return output_path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    }


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


def _string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    columns = _columns(connection, _STRING_TABLE)
    if not {"id", "value"} <= columns:
        return {}
    return {
        int(row["id"]): row["value"]
        for row in connection.execute('SELECT id, value FROM "StringIds"')
    }


def _original_node_ids(connection: sqlite3.Connection) -> dict[int, int]:
    columns = _columns(connection, _NODE_TABLE)
    if "graphNodeId" not in columns or "originalGraphNodeId" not in columns:
        return {}
    result = {}
    query = (
        'SELECT graphNodeId, originalGraphNodeId FROM "CUDA_GRAPH_NODE_EVENTS" '
        "WHERE originalGraphNodeId IS NOT NULL"
    )
    for row in connection.execute(query):
        result[int(row["graphNodeId"])] = int(row["originalGraphNodeId"])
    return result


def _read_graph_kernels(connection, columns, strings):
    optional = [
        name
        for name in (
            "shortName",
            "demangledName",
            "mangledName",
            "deviceId",
            "contextId",
            "streamId",
            "globalPid",
            "graphId",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
        )
        if name in columns
    ]
    selected = ["start", "end", "graphNodeId", *optional]
    query = "SELECT " + ", ".join(map(_quote_identifier, selected))
    query += f" FROM {_quote_identifier(_KERNEL_TABLE)}"
    rows = []
    for row in connection.execute(query):
        raw_name = next(
            (
                row[name]
                for name in ("shortName", "demangledName", "mangledName")
                if name in row.keys() and row[name] is not None
            ),
            None,
        )
        symbol = (
            strings.get(raw_name, str(raw_name)) if raw_name is not None else "unknown"
        )
        rows.append(
            {
                "start_ns": int(row["start"]),
                "duration_ns": max(0, int(row["end"]) - int(row["start"])),
                "graph_node_id": _optional_int(row["graphNodeId"]),
                "graph_id": _row_optional_int(row, "graphId"),
                "global_pid": _row_optional_int(row, "globalPid"),
                "device_id": _row_optional_int(row, "deviceId"),
                "context_id": _row_optional_int(row, "contextId"),
                "stream_id": _row_optional_int(row, "streamId"),
                "symbol": symbol,
                "launch_shape": _launch_shape(row),
            }
        )
    return rows


def _launch_shape(row: sqlite3.Row) -> str | None:
    names = ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")
    if not all(name in row.keys() and row[name] is not None for name in names):
        return None
    grid = "x".join(str(row[name]) for name in names[:3])
    block = "x".join(str(row[name]) for name in names[3:])
    return f"grid={grid};block={block}"


def _canonical_node_id(node_id: int, originals: dict[int, int]) -> int:
    seen = set()
    while node_id in originals and node_id not in seen:
        seen.add(node_id)
        node_id = originals[node_id]
    return node_id


def _source_candidates(symbol_rows, report):
    if report is None:
        return []
    candidates = []
    symbols = [row["symbol"] for row in symbol_rows]
    for implementation in report.get("implementations", []):
        fragments = [
            fragment.lower() for fragment in implementation.get("expected_symbols", [])
        ]
        matching = sorted(
            {
                symbol
                for symbol in symbols
                if any(fragment in symbol.lower() for fragment in fragments)
            }
        )
        if not matching:
            continue
        candidates.append(
            {
                "implementation_id": implementation.get("implementation_id"),
                "implementation": implementation.get("implementation"),
                "scope": implementation.get("scope"),
                "match_basis": "expected_symbol_fragment",
                "matching_symbols": matching,
                "sources": implementation.get("sources", []),
                "cuda_sources": [
                    source
                    for source in implementation.get("sources", [])
                    if Path(source.get("path", "")).suffix in {".cu", ".cuh"}
                ],
                "loaded_libraries": implementation.get("loaded_libraries", []),
            }
        )
    return candidates


def _export_schema_version(connection, tables) -> str | None:
    table = "META_DATA_EXPORT"
    if table not in tables:
        return None
    columns = _columns(connection, table)
    key_column = next((name for name in ("name", "key") if name in columns), None)
    value_column = next((name for name in ("value", "data") if name in columns), None)
    if key_column is None or value_column is None:
        return None
    query = (
        f"SELECT {_quote_identifier(value_column)} FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(key_column)} = ? LIMIT 1"
    )
    row = connection.execute(query, ("EXPORT_SCHEMA_VERSION",)).fetchone()
    return str(row[0]) if row else None


def _load_profile_report(path):
    if path is None:
        return None, None
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"profile report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("implementations"), list):
        raise ValueError(f"{path.name} is not a LangSlanger attribution report")
    return report, path


def _row_optional_int(row: sqlite3.Row, name: str) -> int | None:
    return _optional_int(row[name]) if name in row.keys() else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _add_not_none(values: set[Any], value: Any) -> None:
    if value is not None:
        values.add(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", help="SQLite exported from an Nsight Systems report")
    parser.add_argument(
        "--profile-report", help="LangSlanger profile report JSON for source joins"
    )
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args(argv)
    print(
        write_graph_map(
            args.sqlite,
            profile_report_path=args.profile_report,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
