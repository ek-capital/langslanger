# LangSlanger

> [!CAUTION]
> Experimental research software. Do not use LangSlanger in production.

LangSlanger is Kinetic Capital's performance-first, SGLang-compatible research
fork. It exists to measure inference bottlenecks, test aggressive optimizations,
and preserve enough evidence to map performance changes back to model
architecture and implementation source.

## Compatibility

SGLang remains the compatibility namespace. Existing deployments should keep
using `import sglang`, the `sglang` command, `SGLANG_*` environment variables,
and existing API schemas. LangSlanger adds the `langslanger` command and Python
facade as aliases over the same runtime.

The current compatibility baseline is
[SGLang v0.5.16](https://github.com/sgl-project/sglang/releases/tag/v0.5.16).

LangSlanger-specific flags, configuration fields, and environment variables use
the `langslanger` or `LANGSLANGER_*` namespace.

## Install from source

LangSlanger does not publish a PyPI package. Install the checked-out source in a
dedicated environment:

```bash
git clone https://github.com/ek-capital/langslanger.git
cd langslanger
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./python
```

Run the branded or compatibility command:

```bash
langslanger serve meta-llama/Llama-3.1-8B-Instruct --tp 2
sglang serve meta-llama/Llama-3.1-8B-Instruct --tp 2
```

Container builds are published to
`ghcr.io/ek-capital/langslanger` by the repository's manual image workflow.
Every image is built from the checked-out LangSlanger source and carries its Git
revision in OCI metadata.

## Research workflow

1. Benchmark the real serving path with a pinned model, workload, topology, and
   Git revision.
2. Profile the prefill, decode, or speculative-decoding hot loop.
3. Attribute GPU intervals to stable architectural scopes, implementations,
   symbols, source files, and Git blob hashes.
4. Capture a bounded replay capsule only after a hotspot is identified.
5. Validate the candidate offline, then repeat the serving benchmark on the
   target topology.

See the [LangSlanger documentation](docs_new/index.mdx) for installation,
profiling, and experiment guidance. General SGLang runtime behavior remains
documented by the [upstream SGLang project](https://docs.sglang.io/).

## Contributing

Pull requests must use a Conventional Commit title. Performance changes must
include correctness evidence, raw timing samples, and reproducible baseline and
candidate configurations. See [AGENTS.md](AGENTS.md) for the repository's full
engineering and evidence requirements.

Maintainer: [@evmcheb](https://github.com/evmcheb)
Contact: [contact@ek.capital](mailto:contact@ek.capital)

## Upstream and license

LangSlanger is an independent fork and is not affiliated with or endorsed by
the SGLang project. See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance and
[LICENSE](LICENSE) for license terms.
