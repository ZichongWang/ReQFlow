# Repository Guidelines

## Project Structure & Module Organization
- `models/` holds core network modules (FlowModel, IPA, feature nets).
- `experiments/` provides training/inference entry points (`train_se3_flows.py`, `inference_se3_flows.py`).
- `configs/` contains Hydra YAML configs for datasets, models, training, and inference.
- `openfold/` supplies OpenFold-based utilities and model components.
- `analysis/` includes evaluation scripts and metrics; see `analysis/README.md` for workflows.
- `ProteinMPNN/` is the bundled ProteinMPNN code used during inference.
- `assets/` stores figures for documentation.
- Local-only (ignored by git): `data/`, `ckpts/`, `inference_outputs/`, `outputs/`.

## Build, Test, and Development Commands
Set up the environment and install the package in editable mode:
```bash
conda env create -f reqflow-env.yml
conda activate reqflow-env
pip install -e .
```
Common runs:
```bash
python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional
python -W ignore experiments/train_se3_flows.py -cn train_pdb_base
```
`-cn` selects a config in `configs/`. Update paths such as `ckpt_path` and `inference_subdir` in `configs/inference_unconditional.yaml` before running.

## Coding Style & Naming Conventions
- Python follows 4-space indentation and snake_case for functions/variables (match surrounding files in `models/` and `experiments/`).
- Config keys are snake_case in YAML; keep naming aligned with existing configs.
- No repo-wide formatter is enforced; keep changes consistent with nearby code.

## Testing Guidelines
- There is no formal unit-test suite in this repository.
- Validate changes by running a small inference/training pass (e.g., reduce lengths in `configs/inference_unconditional.yaml`) and, if needed, run evaluation scripts in `analysis/`.
- Record the exact command and config used in your PR notes.

## Commit & Pull Request Guidelines
- Recent history favors short imperative messages (e.g., "Update README.md", "Add helix strand plot").
- Keep commit summaries concise; avoid long prefixes unless needed.
- PRs should include: purpose, key commands run, and where outputs were stored.

## Data & Artifact Handling
- Large artifacts are excluded by `.gitignore` (`data/`, `ckpts/`, `*.ckpt`, `inference_outputs/`, `outputs/`).
- Store checkpoints and generated samples outside git, and reference them in documentation or PR descriptions.
