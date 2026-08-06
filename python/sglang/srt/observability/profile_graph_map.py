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

GRAPH_MAP_SCHEMA_VERSION = 2
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
        node = {
            "global_pid": global_pid,
            "rank_label": _rank_for_global_pid(global_pid, report, devices),
            "graph_id": graph_id,
            "canonical_graph_node_id": canonical_node_id,
            "observed_graph_node_ids": sorted(observed_node_ids),
            "kernel_occurrences": len(values),
            "summed_kernel_ms": sum(kernel["duration_ns"] for kernel in values) / 1e6,
            "_kernel_intervals": [
                (
                    (kernel.get("global_pid"), kernel.get("device_id")),
                    kernel["start_ns"],
                    kernel["start_ns"] + kernel["duration_ns"],
                )
                for kernel in values
            ],
            "symbols": symbol_rows,
            "device_ids": sorted(devices),
            "context_ids": sorted(contexts),
            "stream_ids": sorted(streams),
            "launch_shapes": sorted(launch_shapes),
            "source_mapping_status": source_mapping_status,
            "source_candidates": source_candidates,
        }
        if len(source_candidates) == 1:
            node["resolved_scope"] = source_candidates[0]["scope"]
            node["resolved_implementation"] = source_candidates[0]["implementation"]
        else:
            node["resolved_scope"] = None
            node["resolved_implementation"] = None
        nodes.append(node)

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

    graph_kernel_ms = sum(node["summed_kernel_ms"] for node in nodes)
    resolved_graph_kernel_ms = sum(
        node["summed_kernel_ms"]
        for node in nodes
        if node["source_mapping_status"] == "single_candidate"
    )
    graph_nonoverlap_ms = (
        _interval_union_by_lane(
            interval for node in nodes for interval in node["_kernel_intervals"]
        )
        / 1e6
    )
    interval_partition = _partition_intervals_by_scope(nodes)
    resolved_graph_nonoverlap_ms = interval_partition["resolved_ns"] / 1e6
    components = _component_report(
        nodes,
        graph_kernel_ms,
        graph_nonoverlap_ms,
        interval_partition["exclusive_scope_ns"],
    )
    for node in nodes:
        node.pop("_kernel_intervals", None)
    result = {
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
            "resolved_graph_nodes": sum(
                node["source_mapping_status"] == "single_candidate" for node in nodes
            ),
            "summed_graph_kernel_ms": graph_kernel_ms,
            "resolved_graph_kernel_ms": resolved_graph_kernel_ms,
            "nonoverlap_graph_kernel_ms": graph_nonoverlap_ms,
            "resolved_nonoverlap_graph_kernel_ms": resolved_graph_nonoverlap_ms,
            "cross_component_overlap_ms": (
                interval_partition["cross_component_overlap_ns"] / 1e6
            ),
            "unresolved_nonoverlap_graph_kernel_ms": (
                interval_partition["unresolved_ns"] / 1e6
            ),
            "graph_duration_attribution_ratio": (
                resolved_graph_nonoverlap_ms / graph_nonoverlap_ms
                if graph_nonoverlap_ms
                else None
            ),
        },
        "components": components,
        "graph_nodes": nodes,
        "warnings": warnings,
    }
    result["validation"] = _validate_graph_map(result, report)
    return result


def write_graph_map(
    sqlite_path: str | Path,
    *,
    profile_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    strict: bool = True,
) -> Path:
    sqlite_path = Path(sqlite_path).expanduser().resolve()
    if sqlite_path.name.endswith(".nsys-rep"):
        from sglang.srt.observability.profile_nsys import export_nsys_sqlite

        sqlite_path = export_nsys_sqlite(sqlite_path)
    result = analyze_graph_sqlite(sqlite_path, profile_report_path)
    if strict and result["validation"]["status"] == "failed":
        raise ValueError(
            "graph map validation failed: " + "; ".join(result["validation"]["errors"])
        )
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
    candidates = {}
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
        candidate = {
            "implementation_ids": [implementation.get("implementation_id")],
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
        # Target/draft workers and ranks may declare the same implementation
        # with different runtime conditions.  They are one source candidate,
        # not an ambiguity in the graph-node join.
        key = json.dumps(
            {
                "implementation": candidate["implementation"],
                "scope": candidate["scope"],
                "sources": candidate["sources"],
                "loaded_libraries": candidate["loaded_libraries"],
            },
            sort_keys=True,
        )
        if key in candidates:
            candidates[key]["implementation_ids"] = sorted(
                set(candidates[key]["implementation_ids"])
                | set(candidate["implementation_ids"])
            )
            candidates[key]["matching_symbols"] = sorted(
                set(candidates[key]["matching_symbols"])
                | set(candidate["matching_symbols"])
            )
        else:
            candidates[key] = candidate
    return [candidates[key] for key in sorted(candidates)]


def _component_report(
    nodes: list[dict[str, Any]],
    total_graph_kernel_ms: float,
    total_graph_nonoverlap_ms: float,
    exclusive_scope_ns: dict[str, int],
) -> list[dict[str, Any]]:
    rank_totals = defaultdict(float)
    rank_nodes = defaultdict(list)
    for node in nodes:
        rank = node.get("rank_label") or f"pid:{node.get('global_pid')}"
        rank_totals[rank] += node["summed_kernel_ms"]
        rank_nodes[rank].append(node)
    rank_nonoverlap_totals = {
        rank: _interval_union_by_lane(
            interval for node in values for interval in node["_kernel_intervals"]
        )
        / 1e6
        for rank, values in rank_nodes.items()
    }
    rank_partitions = {
        rank: _partition_intervals_by_scope(values)
        for rank, values in rank_nodes.items()
    }
    grouped = defaultdict(list)
    for node in nodes:
        if node["resolved_scope"] is not None:
            grouped[node["resolved_scope"]].append(node)
    components = []
    for scope, scope_nodes in sorted(grouped.items()):
        summed_ms = sum(node["summed_kernel_ms"] for node in scope_nodes)
        nonoverlap_ms = (
            _interval_union_by_lane(
                interval
                for node in scope_nodes
                for interval in node["_kernel_intervals"]
            )
            / 1e6
        )
        exclusive_nonoverlap_ms = exclusive_scope_ns.get(scope, 0) / 1e6
        candidates = [node["source_candidates"][0] for node in scope_nodes]
        rank_work = defaultdict(float)
        component_rank_nodes = defaultdict(list)
        for node in scope_nodes:
            rank = node.get("rank_label") or f"pid:{node.get('global_pid')}"
            rank_work[rank] += node["summed_kernel_ms"]
            component_rank_nodes[rank].append(node)
        rank_rows = []
        for rank, duration in sorted(rank_work.items()):
            interval_union_ms = (
                _interval_union_by_lane(
                    interval
                    for node in component_rank_nodes[rank]
                    for interval in node["_kernel_intervals"]
                )
                / 1e6
            )
            exclusive_ms = (
                rank_partitions[rank]["exclusive_scope_ns"].get(scope, 0) / 1e6
            )
            rank_rows.append(
                {
                    "rank_label": rank,
                    "summed_kernel_ms": duration,
                    "interval_union_kernel_ms": interval_union_ms,
                    "exclusive_nonoverlap_kernel_ms": exclusive_ms,
                    "percent_of_rank_graph_kernel_work": (
                        100.0 * duration / rank_totals[rank]
                        if rank_totals[rank]
                        else None
                    ),
                    "percent_of_rank_graph_nonoverlap_kernel_time": (
                        100.0 * exclusive_ms / rank_nonoverlap_totals[rank]
                        if rank_nonoverlap_totals[rank]
                        else None
                    ),
                }
            )
        components.append(
            {
                "scope": scope,
                "summed_kernel_ms": summed_ms,
                "interval_union_kernel_ms": nonoverlap_ms,
                "exclusive_nonoverlap_kernel_ms": exclusive_nonoverlap_ms,
                "percent_of_graph_kernel_work": (
                    100.0 * summed_ms / total_graph_kernel_ms
                    if total_graph_kernel_ms
                    else None
                ),
                "percent_of_graph_nonoverlap_kernel_time": (
                    100.0 * exclusive_nonoverlap_ms / total_graph_nonoverlap_ms
                    if total_graph_nonoverlap_ms
                    else None
                ),
                "canonical_graph_nodes": len(scope_nodes),
                "ranks": rank_rows,
                "implementations": sorted(
                    {candidate["implementation"] for candidate in candidates}
                ),
                "symbols": sorted(
                    {
                        symbol["symbol"]
                        for node in scope_nodes
                        for symbol in node["symbols"]
                    }
                ),
                "sources": _unique_records(
                    source
                    for candidate in candidates
                    for source in candidate["sources"]
                ),
                "loaded_libraries": _unique_records(
                    library
                    for candidate in candidates
                    for library in candidate["loaded_libraries"]
                ),
            }
        )
    return components


def _unique_records(records) -> list[dict[str, Any]]:
    unique = {}
    for record in records:
        unique[json.dumps(record, sort_keys=True)] = record
    return [unique[key] for key in sorted(unique)]


def _interval_union_by_lane(intervals) -> int:
    by_lane = defaultdict(list)
    for lane, start, end in intervals:
        if end > start:
            by_lane[lane].append((start, end))
    total = 0
    for lane_intervals in by_lane.values():
        current_start = current_end = None
        for start, end in sorted(lane_intervals):
            if current_end is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        if current_end is not None:
            total += current_end - current_start
    return total


def _partition_intervals_by_scope(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    events_by_lane = defaultdict(list)
    for node in nodes:
        scope = node.get("resolved_scope")
        for lane, start, end in node["_kernel_intervals"]:
            if end > start:
                events_by_lane[lane].append((start, 1, scope))
                events_by_lane[lane].append((end, -1, scope))

    exclusive_scope_ns = defaultdict(int)
    total_ns = resolved_ns = unresolved_ns = cross_component_overlap_ns = 0
    for events in events_by_lane.values():
        active = defaultdict(int)
        previous = None
        index = 0
        events.sort(key=lambda event: event[0])
        while index < len(events):
            timestamp = events[index][0]
            if previous is not None and timestamp > previous:
                duration = timestamp - previous
                active_scopes = {scope for scope, count in active.items() if count > 0}
                if active_scopes:
                    total_ns += duration
                    if None in active_scopes:
                        unresolved_ns += duration
                    else:
                        resolved_ns += duration
                        if len(active_scopes) == 1:
                            exclusive_scope_ns[next(iter(active_scopes))] += duration
                        else:
                            cross_component_overlap_ns += duration
            while index < len(events) and events[index][0] == timestamp:
                _, delta, scope = events[index]
                active[scope] += delta
                index += 1
            previous = timestamp
    return {
        "total_ns": total_ns,
        "resolved_ns": resolved_ns,
        "unresolved_ns": unresolved_ns,
        "cross_component_overlap_ns": cross_component_overlap_ns,
        "exclusive_scope_ns": dict(exclusive_scope_ns),
    }


def _rank_for_global_pid(
    global_pid: int | None, report, device_ids: set[int] | None = None
) -> str | None:
    if report is None:
        return None
    by_pid = {
        process.get("pid"): process.get("rank_label")
        for process in report.get("processes", [])
        if process.get("pid") is not None
    }
    # Nsight encodes a CUDA global process/thread ID as pid << 24 on some
    # export versions; other versions expose the host pid directly.
    if global_pid is not None:
        direct = by_pid.get(global_pid) or by_pid.get(global_pid >> 24)
        if direct is not None:
            return direct

    # Timing and graph attribution are intentionally separate runs, so their
    # host PIDs normally differ. On a single-node run, CUDA device id uniquely
    # identifies the timing report's local rank. Do not guess when the report
    # spans nodes and more than one process matches the device.
    if device_ids and len(device_ids) == 1:
        (device_id,) = device_ids
        matches = [
            process.get("rank_label")
            for process in report.get("processes", [])
            if process.get("local_rank") == device_id
            and process.get("rank_label") is not None
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _validate_graph_map(result, report):
    contracts = report.get("contracts", []) if report else []
    if not contracts:
        return {
            "status": "not_required",
            "errors": [],
            "evidence": "no model profiling contract was supplied",
        }
    required = {
        scope
        for contract in contracts
        for scope in contract.get("graph_required_scopes", [])
    }
    observed = {component["scope"] for component in result["components"]}
    errors = []
    missing = sorted(required - observed)
    if missing:
        errors.append("missing graph components: " + ", ".join(missing))
    minimum = max(
        contract.get("minimum_graph_duration_attribution", 0.0)
        for contract in contracts
    )
    ratio = result["coverage"]["graph_duration_attribution_ratio"]
    if ratio is None or ratio < minimum:
        errors.append(
            f"graph duration attribution below threshold: {ratio!r} < {minimum}"
        )
    if any(contract.get("require_hashed_sources", True) for contract in contracts):
        unhashed = [
            component["scope"]
            for component in result["components"]
            if not component["sources"]
            or any(not source.get("sha256") for source in component["sources"])
        ]
        if unhashed:
            errors.append(
                "graph components lack hashed sources: " + ", ".join(unhashed)
            )
    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "required_scopes": sorted(required),
    }


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
    parser.add_argument(
        "sqlite", help="Nsight Systems .nsys-rep or exported SQLite report"
    )
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
