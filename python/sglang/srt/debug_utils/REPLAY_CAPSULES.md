# Replay capsules

Replay capsules extract one explicitly declared operation boundary for local,
offline correctness checks and microbenchmarks. They are not profiler artifacts:
cloning, device synchronization, host transfer, and file I/O perturb the capture
run, and the manifest marks its timings as non-authoritative.

## Capture contract

Capture is off by default. Both of these variables are required to arm it:

```bash
export SGLANG_REPLAY_CAPTURE_DIR=/local/fast/storage/replays
export SGLANG_REPLAY_CAPTURE_SCOPES=model.attention.mla
```

The optional bounds are
`SGLANG_REPLAY_CAPTURE_RANKS` (default `0`),
`SGLANG_REPLAY_CAPTURE_MAX_CASES_PER_SCOPE_RANK` (default `1`),
`SGLANG_REPLAY_CAPTURE_MAX_TOTAL_CASES` (default `8`), and
`SGLANG_REPLAY_CAPTURE_MAX_BYTES` (default 1 GiB). Scope matching is exact.

A target call site uses `begin_replay_capture` immediately before the operation
and `finish` immediately after it. It must name flat tensor mappings for inputs,
read-only state, mutable state before/after, and outputs. It must also declare the
minimum replay topology. The initial clones are enqueued on the calling CUDA
stream before the target operation; serialization happens afterward. Capture is
refused inside CUDA graph capture.

Do not add a capture point speculatively. Add the declaration beside a measured
hotspot so the required tensors and state boundary are explicit in review.

## Artifact

Each completed `*.replay/` directory is transactional and contains:

- `tensors.safetensors`: `input.*`, `read_only.*`, `mutable_before.*`,
  `output.*`, and `mutable_after.*` tensors;
- `manifest.json`: shapes, strides, dtypes, source devices and streams, caller
  metadata, required topology, runtime, source commit, dirty-diff hash, and the
  tensor-file checksum;
- `COMPLETE`: manifest and tensor checksums. Its absence means the capsule is
  incomplete and the runner rejects it.

Capsules may contain prompts, tokens, activations, or proprietary weights. Keep
them on approved local storage; never commit or upload them by default.

## Replay contract

The candidate and optional reference are addressed as `module.path:function`.
They receive `(inputs, read_only_state, mutable_state, metadata)` and return
`(outputs, mutable_state_after)`, both as flat tensor dictionaries. The runner
checks output names, shapes, dtypes and tolerances, checks every declared mutable
state tensor, then times the callable only. Input/state reset happens outside the
timed region. Candidate/reference order alternates when both are supplied.

```bash
python -m sglang.srt.debug_utils.replay_runner \
  /local/fast/storage/replays/<capsule>.replay \
  --candidate package.module:candidate \
  --reference package.module:reference \
  --device cuda \
  --warmup 10 \
  --iterations 100 \
  --output /local/fast/storage/replays/result.json
```

The JSON result preserves every timing sample, correctness tolerances, callable
source hashes, hardware/runtime identity, topology validation, and median paired
speedup. Offline replay timings are diagnostic only; any winning candidate still
requires the same-run serving benchmark and profile on the target GPU topology.
