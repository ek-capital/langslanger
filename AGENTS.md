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

## SGLang drop-in compatibility

Drop-in compatibility with the SGLang release identified in `README.md` is a
hard repository invariant. Replacing an SGLang checkout, wheel, or container
with LangSlanger must not require a user to rewrite an existing command,
configuration file, Python import, deployment manifest, or API client.

The product name is LangSlanger; the compatibility namespace remains SGLang.
Agents MUST:

- preserve the `sglang` Python package, `sglang` executable, `sglang serve`,
  `sglang generate`, `sglang version`, and
  `python -m sglang.launch_server`;
- preserve every upstream option and alias, including its destination, action,
  type, choices, `nargs`, default, deprecation behavior, and precedence between
  command-line and configuration values;
- preserve upstream configuration keys, `SGLANG_*` environment variables,
  metrics, HTTP schemas, and operational defaults;
- implement branded entry points such as `langslanger` as thin aliases over the
  same parser and runtime rather than as a second copied implementation;
- add new server options only under `--langslanger-*`, configuration fields
  under `langslanger_*`, and environment variables under `LANGSLANGER_*`;
- keep profiling, capture, tracing, and other overhead-producing additions
  disabled until explicitly requested;
- extend the upstream `ServerArgs` and launch paths in place; never maintain a
  hand-copied mirror of upstream arguments;
- retain upstream tests unchanged and add LangSlanger-specific tests
  separately; and
- treat a future upstream name collision as owned by upstream, renaming or
  deprecating the LangSlanger addition without changing upstream semantics.

Do not mass-replace `sglang` with `langslanger` in source paths, imports,
protocol names, serialized class paths, CUDA symbols, metrics, or environment
variables. Those names are compatibility interfaces and also keep upstream
diffs reviewable.

Before completing a change to CLI parsing, `ServerArgs`, configuration,
entrypoints, packaging, or public APIs, run the focused LangSlanger identity
tests plus the relevant upstream tests. A compatibility snapshot or baseline
must never be changed merely to silence a failure. Stop and request maintainer
direction before making an intentional compatibility break.

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

## Grugbrain engineering bias

Use [The Grug Brained Developer](https://grugbrain.dev/) as a default
tie-breaker: complexity is a real cost, including complexity added with good
intentions. Keep all required behavior and evidence, but implement them in the
most direct form that remains correct and debuggable.

- Start with the 80/20 solution that answers the present measured question. Do
  not build extension points, registries, services, configuration languages, or
  generalized frameworks for hypothetical future experiments.
- Let stable cut points emerge from real uses before extracting an abstraction.
  Prefer a narrow function and plain data over a class hierarchy or callback
  graph. A little obvious duplication is often cheaper than a premature DRY
  mechanism.
- Keep behavior local to the thing that performs it. A dispatch decision, its
  profiler declaration, and its source attribution should be understandable by
  reading the same execution path.
- Prefer linear control flow, named intermediate values, and explicit state.
  Avoid dense expressions, clever metaprogramming, unnecessary generics, and
  hidden global behavior when ordinary code will do.
- Understand an existing fence before removing or redesigning it. Make small
  refactors that leave the repository working after each step, and avoid broad
  cleanup around a focused experiment.
- Use tools and structured logging to make runtime behavior visible. Carry a
  profile, request, or iteration identifier through related evidence, and keep
  diagnostic collection dynamically controllable.
- Test the real cut point. Favor focused integration tests, add a regression
  test before fixing a reproduced bug, keep end-to-end coverage small and
  valuable, and mock only where the real boundary is impractical.
- Treat concurrency and distributed state as complexity multipliers. Reuse the
  runtime's existing ownership and synchronization model; do not add background
  coordination or shared mutable state without a measured need.

Simplicity is not permission to weaken evidence. Do not drop ranks, conflate
overlapping GPU work with elapsed time, guess a kernel-to-source mapping, or
substitute a microbenchmark for serving validation merely because doing so is
easier. When the simple implementation cannot prove a claim, report the result
as unknown and add only the smallest missing measurement needed to resolve it.

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

### Canonical prompt workloads

Use the two published `nvidia/SPEED-Bench` suites as LangSlanger's canonical
semantic prompt categories:

- `speed-bench-qualitative` measures speculative-decoding acceptance and
  latency across the eleven semantic categories, preserving multi-turn cases;
- `speed-bench-throughput` measures prefill, decode, batching, and speculative
  throughput across the fixed 1K, 2K, 8K, 16K, and 32K input-length artifacts
  and their low-, mixed-, and high-entropy categories.

Treat `sharegpt`, `random`, `random-ids`, generated shared prefixes, and custom
prompt files as diagnostic or workload-specific controls, not canonical
semantic results. A scoped experiment may run only the relevant SPEED-Bench
suite, but a general performance claim should report both. Always record the
dataset repository and revision, materialized artifact SHA-256, suite/config,
category filter, sampling seed, output length, request rate, concurrency, and
chat template or tokenizer revision. Never benchmark raw source-placeholder
rows from the Hugging Face export.

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
