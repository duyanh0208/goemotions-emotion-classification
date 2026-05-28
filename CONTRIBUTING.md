# Contributing Guide

Development conventions for this project.

---

## 1. Branch Naming

| Branch type | Pattern | Example |
|-------------|---------|---------|
| Stable production | `main` | — |
| Active development | `develop` | — |
| New feature / module | `feature/<name>` | `feature/lr-scheduler` |
| New experiment run | `experiment/<name>` | `experiment/roberta-large` |
| Bug fix | `fix/<name>` | `fix/tokenizer-overflow` |
| Documentation only | `docs/<name>` | `docs/update-setup` |

**Rule:** Never commit directly to `main`. Branch from `develop`, work there, then open a PR back into `develop`. `develop` is merged into `main` only at milestone releases.

```
main
 └── develop
      ├── feature/add-deberta
      ├── experiment/gemini-fewshot-k10
      └── fix/nan-loss-on-empty-batch
```

---

## 2. Commit Message Format

```
<type>: <short description (≤72 chars)>

[optional body — explain WHY, not WHAT]
```

| Type | When to use |
|------|-------------|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `exp` | Experiment run or result update |
| `refactor` | Code restructure (no behaviour change) |
| `test` | Adding or updating tests |
| `chore` | Config, CI, tooling, dependency updates |

**Examples:**
```
feat: add per-class threshold optimisation in evaluate.py

exp: run roberta_base full training — F1-macro 0.51

fix: handle empty label list in _emotions_to_multihot

docs: update EXPERIMENTS.md with bert_base results

refactor: extract checkpoint logic into _save_checkpoint helper
```

---

## 3. Pull Request Process

1. **Branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature
   ```

2. **Develop & test** locally — run the suite before pushing:
   ```bash
   pytest tests/ -v
   ```

3. **Lint** (must pass CI):
   ```bash
   black src/ scripts/ tests/ --line-length 100
   flake8 src/ scripts/ tests/ --max-line-length 100
   ```

4. **Push** and open a PR against `develop`:
   ```bash
   git push origin feature/your-feature
   # then open PR on GitHub
   ```

5. **PR description** should include:
   - What changed and why
   - Test results or sample output
   - Link to relevant config / EXPERIMENTS.md entry

6. **Merge strategy**: squash-and-merge for feature branches; regular merge for experiment branches (to preserve run history).

---

## 4. Code Style

| Rule | Value |
|------|-------|
| Formatter | `black` |
| Linter | `flake8` |
| Max line length | 100 |
| Imports order | `isort` (stdlib → third-party → local) |
| Docstrings | Google style |
| Type hints | Required for all public functions |
| Naming — files/vars | `snake_case` |
| Naming — classes | `PascalCase` |
| Naming — constants | `UPPER_SNAKE_CASE` |

Install dev tools:
```bash
pip install black flake8 isort pytest
```

Run all checks at once:
```bash
black src/ scripts/ tests/ --line-length 100 --check
flake8 src/ scripts/ tests/ --max-line-length 100
isort src/ scripts/ tests/ --check-only
```

---

## 5. Adding a New Experiment

### Step 1 — Create a config file
```bash
cp configs/bert_base.yaml configs/my_experiment.yaml
# Edit: experiment.name, model.name, hyperparameters, paths
```

### Step 2 — Run the experiment
```bash
# BERT / RoBERTa fine-tuning
python -m src.train --config configs/my_experiment.yaml

# LLM inference
python -m src.llm_inference --config configs/my_experiment.yaml
```

Results are automatically saved to `results/metrics/<experiment_name>_metrics.json`.

### Step 3 — Update EXPERIMENTS.md

Open `docs/EXPERIMENTS.md` and fill in a new entry (copy an existing template block):

```markdown
## EXP-05: my_experiment

| Field | Value |
|-------|-------|
| Date | 2026-MM-DD |
| Config | `configs/my_experiment.yaml` |
| Hardware | ... |
| Training time | ... |

### Results
| Metric | Value |
|--------|-------|
| F1-macro | 0.XXX |
```

### Step 4 — Regenerate comparison table
```bash
python scripts/compare_results.py
```

### Step 5 — Commit
```bash
git add configs/my_experiment.yaml docs/EXPERIMENTS.md results/metrics/
git commit -m "exp: add my_experiment — F1-macro 0.XXX"
```

---

## 6. Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 7. Environment & Secrets

- **Never commit `.env`** — it is in `.gitignore`.
- Use `.env.example` as the template; update it when you add new env vars.
- Required keys: `GEMINI_API_KEY`, `WANDB_API_KEY` (optional).

---

## 8. Data & Models Policy

- Raw data (`data/`) and model checkpoints (`results/models/`) are **gitignored** — do not force-add them.
- Metrics JSONs and plots in `results/metrics/` and `results/plots/` **are** committed (small files, important for reproducibility).

---

## 9. Code Review Checklist

Before merging:
- [ ] Code runs (smoke test)
- [ ] `pytest tests/` passes
- [ ] No hardcoded paths or API keys
- [ ] No commented-out dead code
- [ ] Type hints + docstrings on all public functions
- [ ] `EXPERIMENTS.md` updated (if this is an experiment branch)
- [ ] `black` + `flake8` pass

---

## 10. Questions & Issues

Open a GitHub Issue with label `question` or `bug`. Include:
- Minimal reproducible example
- Full error traceback
- Python / CUDA version (`python --version`, `nvidia-smi`)
