# LangSlanger agent instructions

## 1. Commit messages: Conventional Commits are required

Every commit created for this repository MUST follow
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```

Use these types unless a more specific established type is clearly better:

- `feat`: add a user-visible capability
- `fix`: correct faulty behavior
- `perf`: improve performance without changing intended behavior
- `refactor`: restructure code without changing behavior or performance intent
- `test`: add or correct tests
- `docs`: change documentation only
- `build`: change build code or dependencies
- `ci`: change continuous integration
- `chore`: repository maintenance not covered above
- `revert`: revert earlier work

Scopes are optional, short nouns describing the affected subsystem, for example
`profiling`, `replay`, `scheduler`, `spec`, `kernel`, `model`, `docs`, or `ci`.

Examples:

```text
perf(profiling): reduce disabled scope overhead
feat(replay): export one-shot MLA replay capsules
fix(spec): preserve committed-token counts across partial rounds
docs: explain graph-on and mapping profiles
feat(api)!: replace the profiler request format
```

Breaking changes MUST use `!` in the prefix or a `BREAKING CHANGE:` footer. Keep
each commit focused on one logical change. If a change needs more than one type,
prefer separate commits. Inspect the staged diff before committing and never
include unrelated user changes. PR titles SHOULD use the same form when the
repository uses squash merging.

## Project identity

LangSlanger is Kinetic Capital's experimental, performance-first,
SGLang-compatible research fork. It is deliberately not production software.
Preserve SGLang API and operational compatibility unless an experiment
explicitly requires a divergence, and document every intentional divergence.

Keep LangSlanger changes easy to distinguish from upstream SGLang. Avoid
unrelated cleanup in performance patches, retain upstream structure where
practical, and make future rebases or cherry-picks straightforward.

## Development style

- Prefer the smallest implementation that answers a measured need.
- Do not introduce an abstraction before at least two concrete uses reveal a
  stable boundary.
- Keep behavior beside the code that performs it. In particular, profiler scope
  declarations and implementation/source attribution belong beside the relevant
  execution or dispatch path.
- A little obvious repetition is preferable to hidden state, registration magic,
  or a premature plugin framework.
- Make small changes that leave the repository working. Do not combine broad
  refactors, instrumentation, analysis, and optimization in one patch.
- Respect existing code until its purpose and tests are understood. Preserve
  unrelated working-tree changes.

## Performance work

Do not optimize from intuition alone. Begin with a concrete profile or benchmark
that identifies a specific bottleneck.

Every performance claim MUST include:

- the baseline and candidate Git commits;
- exact model, weights revision, hardware, topology, software versions, launch
  arguments, environment, and workload;
- correctness checks appropriate to the affected boundary;
- warmup and measurement procedures;
- raw samples or a durable artifact containing them;
- latency and throughput metrics with their precise normalization;
- any memory, startup, accuracy, or tail-latency tradeoff;
- the observed variance and whether the change is larger than measurement noise.

Compare baseline and candidate on the same host and configuration whenever
possible. Prefer alternating paired runs over comparisons with historical
numbers. Microbenchmarks are diagnostic evidence; validate meaningful wins in
the closest practical end-to-end serving path.

Never describe summed GPU kernel work as wall-clock latency when streams can
overlap. Keep at least these quantities distinct:

- request latency;
- request-weighted milliseconds per committed token;
- GPU elapsed or interval-union time;
- summed GPU work;
- GPU service cost per token;
- compute/communication overlap;
- cross-rank skew.

## Profiling architecture

Build on SGLang's existing profiler manager and `/start_profile` and
`/stop_profile` controls. Prefer a few explicit primitives over a new profiler
framework:

- `profile_scope(name, ...)` for stable semantic execution boundaries;
- `record_profile_step(...)` for workload and normalization context;
- `record_profile_impl(...)` for the implementation selected by a dispatcher.

Semantic scope names MUST be stable and low-cardinality. Put dynamic values such
as rank, layer, batch bucket, token counts, and graph key in structured metadata
or sidecars rather than inventing a new scope name for every invocation.

Profiles SHOULD make it possible to join:

```text
forward and workload bucket
-> semantic architecture scope
-> observed kernel or CUDA-graph node
-> selected implementation and symbol
-> source file and Git blob hash
```

Source attribution MUST come from explicit execution and dispatch declarations
where possible. Kernel-name regular expressions are diagnostic fallbacks only.
Reports must state their timing basis, attribution source, coverage, and
unattributed residual. If evidence is insufficient, fail closed instead of
inventing a precise architectural breakdown.

Collect and preserve evidence from all participating ranks. Do not call the rank
with the largest sum of kernel durations the distributed critical path without
additional timing evidence.

## Replay capture

Profiling and tensor replay capture are separate passes. Capture changes
execution through cloning, synchronization, memory pressure, and I/O, so capture
timings are never performance evidence.

Replay capture MUST be:

- explicitly armed and targeted at a named semantic scope;
- one-shot or otherwise bounded by required case and byte limits;
- written first to suitable local storage;
- transactional, with a completion marker and checksums;
- explicit about inputs, read-only state, mutable state, outputs, metadata, and
  required topology;
- validated against reference outputs and state mutations before benchmarking a
  candidate implementation.

Clone inputs before optimized code can mutate or reuse their storage. Respect the
producer CUDA stream before asynchronous copies. Start with local, single-rank
replay capsules; add distributed or stateful machinery only for a demonstrated
hotspot that requires it.

## Testing and validation

Use the lightest test that exercises the real boundary. Favor integration tests
for profiler-manager behavior, trace-to-report analysis, dispatch attribution,
and replay correctness. Add focused unit tests for interval arithmetic,
normalization, serialization, and other deterministic logic. Keep a small number
of end-to-end GPU tests for the most important workflows.

Performance instrumentation must test both enabled and disabled paths. Verify
that disabled instrumentation does not synchronize the GPU, inspect tensors,
write artifacts, or measurably affect normal execution.

Follow the existing SGLang test registration and formatting conventions in
`test/README.md` and the relevant local documentation. Run focused checks first;
expand validation in proportion to the affected hardware, model, and distributed
surface. Do not claim hardware integration from source checks or mocked tests.

## Documentation and nested instructions

Record exact commands, assumptions, limitations, and artifact formats for
experimental workflows. Do not document flags, defaults, or behavior from
memory; verify them against the checked-out code and the tested environment.

More-specific `AGENTS.md` files override this file within their directory tree.
In particular, follow `docs_new/AGENTS.md` for documentation-site work in the
current stable tree.

## Safety and repository hygiene

- Never commit secrets, credentials, model weights, raw user prompts, captured
  activations containing sensitive data, profiler traces, or large replay
  artifacts.
- Store large artifacts outside Git and commit only small manifests, checksums,
  schemas, and documentation needed to reproduce them.
- Do not run destructive Git commands, discard unrelated changes, or rewrite
  shared history without explicit authorization.
- Treat code submitted for GPU execution as untrusted until reviewed. Use
  isolated, resource-bounded environments for external or generated kernels.
