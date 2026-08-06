import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from sglang.srt.observability.profile_graph_map import (
    analyze_graph_sqlite,
    write_graph_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestProfileGraphMap(unittest.TestCase):
    def test_maps_cloned_graph_nodes_to_symbols_and_declared_sources(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            sqlite_path = directory / "run.sqlite"
            with sqlite3.connect(sqlite_path) as connection:
                connection.executescript("""
                    CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
                    CREATE TABLE CUDA_GRAPH_NODE_EVENTS (
                        graphNodeId INTEGER NOT NULL,
                        originalGraphNodeId INTEGER
                    );
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                        start INTEGER NOT NULL,
                        end INTEGER NOT NULL,
                        deviceId INTEGER NOT NULL,
                        contextId INTEGER NOT NULL,
                        streamId INTEGER NOT NULL,
                        globalPid INTEGER,
                        shortName INTEGER NOT NULL,
                        graphNodeId INTEGER,
                        gridX INTEGER,
                        gridY INTEGER,
                        gridZ INTEGER,
                        blockX INTEGER,
                        blockY INTEGER,
                        blockZ INTEGER
                    );
                    """)
                connection.execute(
                    "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
                    ("EXPORT_SCHEMA_VERSION", "3.28.1"),
                )
                connection.execute(
                    "INSERT INTO StringIds VALUES (?, ?)",
                    (1, "void mla_kernel<float>()"),
                )
                connection.executemany(
                    "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?, ?)",
                    [(10, None), (11, 10)],
                )
                kernel_row = (
                    100,
                    200,
                    6,
                    2,
                    9,
                    42,
                    1,
                    11,
                    1,
                    1,
                    1,
                    128,
                    1,
                    1,
                )
                connection.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [kernel_row, (300, 500, *kernel_row[2:])],
                )

            profile_report_path = directory / "profile-report.json"
            profile_report_path.write_text(
                json.dumps(
                    {
                        "profile_id": "profile-1",
                        "processes": [{"pid": 42, "rank_label": "TP-6"}],
                        "implementations": [
                            {
                                "implementation_id": "impl-1",
                                "implementation": "mla.test",
                                "scope": "model.attention.mla",
                                "expected_symbols": ["mla_kernel"],
                                "sources": [
                                    {
                                        "path": "kernel.cu",
                                        "sha256": "source-sha",
                                        "git_blob": "source-blob",
                                    }
                                ],
                                "loaded_libraries": [],
                            }
                        ],
                        "contracts": [
                            {
                                "contract_id": "contract-1",
                                "graph_required_scopes": ["model.attention.mla"],
                                "minimum_graph_duration_attribution": 0.8,
                                "require_hashed_sources": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = analyze_graph_sqlite(sqlite_path, profile_report_path)
            self.assertFalse(result["authoritative_serving_timing"])
            self.assertEqual(result["input"]["nsight_export_schema_version"], "3.28.1")
            self.assertEqual(result["coverage"]["canonical_graph_nodes"], 1)
            self.assertEqual(
                result["coverage"]["graph_duration_attribution_ratio"], 1.0
            )
            self.assertEqual(result["coverage"]["nonoverlap_graph_kernel_ms"], 0.0003)
            node = result["graph_nodes"][0]
            self.assertEqual(node["canonical_graph_node_id"], 10)
            self.assertEqual(node["rank_label"], "TP-6")
            self.assertEqual(node["observed_graph_node_ids"], [11])
            self.assertEqual(node["kernel_occurrences"], 2)
            self.assertEqual(node["symbols"][0]["symbol"], "void mla_kernel<float>()")
            self.assertEqual(node["source_mapping_status"], "single_candidate")
            self.assertEqual(
                node["source_candidates"][0]["sources"][0]["git_blob"],
                "source-blob",
            )
            self.assertEqual(result["components"][0]["scope"], "model.attention.mla")
            self.assertEqual(
                result["components"][0]["percent_of_graph_nonoverlap_kernel_time"],
                100.0,
            )
            self.assertEqual(result["components"][0]["ranks"][0]["rank_label"], "TP-6")
            self.assertEqual(result["validation"]["status"], "passed")

            output_path = write_graph_map(
                sqlite_path, profile_report_path=profile_report_path
            )
            self.assertEqual(
                json.loads(output_path.read_text())["input"]["profile_id"],
                "profile-1",
            )

    def test_rejects_export_without_graph_node_column(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            sqlite_path = Path(temporary_dir) / "run.sqlite"
            with sqlite3.connect(sqlite_path) as connection:
                connection.execute(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                    "(start INTEGER NOT NULL, end INTEGER NOT NULL)"
                )
            with self.assertRaisesRegex(ValueError, "graphNodeId"):
                analyze_graph_sqlite(sqlite_path)


if __name__ == "__main__":
    unittest.main()
