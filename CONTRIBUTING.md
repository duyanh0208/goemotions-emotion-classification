# Contributing Guide

Quy ước phát triển cho project này.

---

## Git Workflow

### Branch naming
- `main` — code stable, ready to ship
- `develop` — development branch (default)
- `feature/<name>` — tính năng mới (e.g., `feature/llm-fewshot`)
- `experiment/<name>` — experiments (e.g., `experiment/roberta-large`)
- `fix/<name>` — bug fixes
- `docs/<name>` — chỉ update docs

### Commit Messages

Format: `<type>: <description>`

Types:
- `feat:` Tính năng mới
- `fix:` Bug fix
- `docs:` Cập nhật docs
- `refactor:` Refactor code (không đổi behavior)
- `test:` Thêm/sửa tests
- `chore:` Build, deps, configs
- `exp:` Experiment runs

Examples:
```
feat: add LoRA fine-tuning support
fix: handle empty labels in dataset
docs: update SETUP.md with Windows-specific notes
exp: run BERT-base with lr=3e-5
refactor: extract model class into separate file
```

### Pull Request Process

1. Tạo branch từ `develop`
2. Make changes + commits
3. Push branch
4. Tạo PR vào `develop`
5. Self-review trước khi merge
6. Merge sau khi CI pass

---

## Code Style

### Python
- **PEP 8** + max line length 100
- Use **type hints** cho function signatures
- Use **docstrings** (Google style)
- Use **`black`** formatter trước commit

```bash
black src/ scripts/
flake8 src/ scripts/ --max-line-length=100
```

### Naming Conventions
- Files/modules: `snake_case`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

---

## Adding a New Experiment

1. Tạo config YAML mới trong `configs/`
2. Add description vào `docs/EXPERIMENTS.md`
3. Run experiment, log W&B
4. Update results trong EXPERIMENTS.md
5. Commit config + results JSON

---

## Documentation

- Cập nhật README.md khi có major changes
- Cập nhật EXPERIMENTS.md sau mỗi run
- Inline comments cho logic phức tạp
- Tiếng Việt OK cho docs, tiếng Anh cho code

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data.py -v
```

---

## Code Review Checklist

Trước khi merge:
- [ ] Code chạy được (smoke test)
- [ ] Tests pass
- [ ] No hardcoded paths/keys
- [ ] No commented-out code
- [ ] Type hints + docstrings
- [ ] EXPERIMENTS.md updated (nếu là exp)
