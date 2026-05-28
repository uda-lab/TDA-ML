# Third-party pins

## `pytorch_topological.ref`

Pins the commit checked out under `pytorch-topological/` for the `torch_topological` path dependency in `pyproject.toml`.

- **Fetch / sync:** `./scripts/ensure_pytorch_topological.sh` (used by CI and `README.md`).
- **Bump upstream:** choose a new commit on [aidos-lab/pytorch-topological](https://github.com/aidos-lab/pytorch-topological), update the SHA in `pytorch_topological.ref`, run the ensure script, then `uv sync` and `uv run ruff check .`.

`.gitmodules` may list the same repository; the ref file is the canonical pin when the parent repo does not record a submodule gitlink.
