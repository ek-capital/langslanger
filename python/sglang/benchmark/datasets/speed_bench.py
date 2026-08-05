"""Canonical SPEED-Bench workloads for the SGLang serving benchmark.

LangSlanger treats the two published ``nvidia/SPEED-Bench`` suites as
different workloads:

* ``speed-bench-qualitative`` measures speculative-decoding behaviour across
  semantic categories and preserves all conversation turns.
* ``speed-bench-throughput`` measures serving performance across the published
  fixed-input-length and entropy buckets.

``speed-bench`` remains a compatibility alias for the throughput workload.
The input must be a source-complete JSONL artifact produced by the SPEED-Bench
measurement framework.  Raw Hub rows that still contain source placeholders
are rejected instead of silently benchmarking the placeholder text.
"""

import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

SPEED_BENCH_REPO_ID = "nvidia/SPEED-Bench"
SPEED_BENCH_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"

SPEED_BENCH_QUALITATIVE = "speed-bench-qualitative"
SPEED_BENCH_THROUGHPUT = "speed-bench-throughput"
SPEED_BENCH_LEGACY = "speed-bench"

SPEED_BENCH_SUITES: Dict[str, str] = {
    SPEED_BENCH_QUALITATIVE: "qualitative",
    SPEED_BENCH_THROUGHPUT: "throughput",
    SPEED_BENCH_LEGACY: "throughput",
}

QUALITATIVE_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "coding",
        "humanities",
        "math",
        "multilingual",
        "qa",
        "rag",
        "reasoning",
        "roleplay",
        "stem",
        "summarization",
        "writing",
    }
)
THROUGHPUT_CATEGORIES: FrozenSet[str] = frozenset(
    {"high_entropy", "low_entropy", "mixed"}
)
SUITE_CATEGORIES: Dict[str, FrozenSet[str]] = {
    "qualitative": QUALITATIVE_CATEGORIES,
    "throughput": THROUGHPUT_CATEGORIES,
}

_CHAT_BACKENDS = {"sglang-oai-chat", "vllm-chat", "lmdeploy-chat"}
_SOURCE_PLACEHOLDER = (
    "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
)


@dataclass
class SpeedBenchDataset(BaseDataset):
    dataset_path: str
    suite: str
    category: Optional[str]
    output_len: int
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "SpeedBenchDataset":
        if not args.dataset_path:
            raise ValueError(
                "--dataset-path must point to a materialized SPEED-Bench JSONL "
                "file produced by the SPEED-Bench measurement framework."
            )

        dataset_name = args.dataset_name
        try:
            suite = SPEED_BENCH_SUITES[dataset_name]
        except KeyError as exc:
            names = ", ".join(sorted(SPEED_BENCH_SUITES))
            raise ValueError(
                f"Unsupported SPEED-Bench dataset name {dataset_name!r}; use {names}."
            ) from exc

        backend = getattr(args, "backend", None)
        if suite == "qualitative" and backend not in _CHAT_BACKENDS:
            raise ValueError(
                f"{SPEED_BENCH_QUALITATIVE} preserves multi-turn prompts and "
                f"requires a chat backend: {', '.join(sorted(_CHAT_BACKENDS))}."
            )

        category = getattr(args, "speed_bench_category", None) or None
        allowed_categories = SUITE_CATEGORIES[suite]
        if category and category not in allowed_categories:
            raise ValueError(
                f"Invalid {suite} SPEED-Bench category {category!r}; expected one "
                f"of {', '.join(sorted(allowed_categories))}."
            )

        return cls(
            dataset_path=args.dataset_path,
            suite=suite,
            category=category,
            output_len=getattr(args, "speed_bench_output_len", 512),
            num_requests=args.num_prompts,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        prompt_rows = []
        observed_categories = set()
        with open(self.dataset_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                row = json.loads(line)
                category = row.get("category")
                if category:
                    observed_categories.add(category)
                if self.category and category != self.category:
                    continue

                turns = [turn for turn in row.get("turns", []) if turn]
                if not turns:
                    continue
                if any(_SOURCE_PLACEHOLDER in turn for turn in turns):
                    raise ValueError(
                        f"Unmaterialized SPEED-Bench source placeholder at "
                        f"{self.dataset_path}:{line_number}; build the dataset with "
                        "the SPEED-Bench measurement framework before profiling."
                    )
                prompt_rows.append(turns)

        unexpected_categories = observed_categories - SUITE_CATEGORIES[self.suite]
        if unexpected_categories:
            raise ValueError(
                f"{self.dataset_path} does not match the {self.suite} SPEED-Bench "
                f"suite; unexpected categories: "
                f"{', '.join(sorted(unexpected_categories))}."
            )

        if not prompt_rows:
            raise ValueError(
                f"No rows found in {self.dataset_path}"
                + (f" for category={self.category}" if self.category else "")
            )

        dataset_rows = [self._to_dataset_row(tokenizer, turns) for turns in prompt_rows]

        if self.num_requests <= len(dataset_rows):
            return random.sample(dataset_rows, self.num_requests)

        sampled_rows = dataset_rows * (self.num_requests // len(dataset_rows) + 1)
        sampled_rows = sampled_rows[: self.num_requests]
        random.shuffle(sampled_rows)
        return sampled_rows

    def _to_dataset_row(
        self, tokenizer: PreTrainedTokenizerBase, turns: List[str]
    ) -> DatasetRow:
        if self.suite == "qualitative":
            # bench_serving recognizes List[str] as a multi-turn conversation,
            # including the one-turn case. This keeps the request shape uniform.
            prompt_len = sum(len(tokenizer.encode(turn)) for turn in turns)
            return DatasetRow(
                prompt=turns,
                prompt_len=prompt_len,
                output_len=self.output_len,
            )

        prompt_text = turns[0]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt = tokenizer.decode(prompt_ids)
        except Exception:
            prompt_ids = tokenizer.encode(prompt_text)
            prompt = prompt_text

        return DatasetRow(
            prompt=prompt,
            prompt_len=len(prompt_ids),
            output_len=self.output_len,
        )
