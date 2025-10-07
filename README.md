# 7tAxiom

A modular, Python-based framework for scalable statistical testing and validation. 7tAxiom provides services and engines for running Two One-Sided Tests (TOST) and related linear-model analyses, managing artifacts, and enforcing repeatable deployment and validation workflows.

## Highlights
- Clean separation of concerns: service layer, engines, artifacts, validation
- Deterministic, reproducible runs with clear artifact outputs
- Config-driven execution for local, CI, and production use
- Extensible architecture for adding new tests and models
- Batteries-included validation and deployment guidance

## Repository Structure
- 7dsaService — service layer and orchestration utilities
- ToSTLinearEngine — core linear-model and TOST computation engine
- ARTIFACTS — generated outputs, reports, and serialized objects
- VALIDATION — validation datasets, scenarios, and checks
- DEPLOYMENT.md — deployment patterns and environments
- PRINCIPLES.md — core design principles and philosophy
- STRATEGY.md — high-level roadmap and technical strategy
- TODO.md — short-term tasks and backlog

## Quick Start
Prerequisites:
- Python 3.10+
- pip or uv (recommended)
- make (optional)

Setup:
1) Clone the repo
   git clone https://github.com/AXI0MH1VE/7tAxiom
   cd 7tAxiom
2) Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
3) Install dependencies
   pip install -U pip
   pip install -e .[dev]  # or: pip install -r requirements.txt if provided
4) Run a smoke test
   python -m pytest -q || echo "Run minimal checks if tests are not yet defined"

Minimal usage example (pseudo-code):
```python
from ToSTLinearEngine import tost

# x, y are arrays or pandas Series
result = tost.run(x, y, equivalence_margin=(-0.2, 0.2))
print(result.summary())
```

## Features
- TOST and linear-model equivalence testing utilities
- Configurable pipelines for experiment runs
- Artifact management for metrics, plots, and model objects
- Validation harness to verify assumptions and guard against regressions
- CLI-friendly entry points for automation (planned)

## Configuration
- Environment variables and/or YAML/JSON config files to define:
  - data sources and paths
  - equivalence margins and model parameters
  - output directories for artifacts
  - run IDs and reproducibility seeds

## Onboarding
- Read PRINCIPLES.md for design intent
- Review STRATEGY.md for big-picture direction
- Use DEPLOYMENT.md to align your environment (local/CI)
- Explore VALIDATION/ to see how scenarios and checks are organized
- Start experiments using ToSTLinearEngine as a reference implementation

## Development
- Style: black, isort, flake8 (or ruff)
- Testing: pytest with coverage
- Lint/Format: pre-commit hooks recommended

Common commands:
```
make install       # install dev deps
make lint          # run linters/formatters
make test          # run unit tests
make validate      # run validation scenarios
```

Without make:
```
pip install -e .[dev]
pytest -q
```

## Contributing
- Fork the repo and create feature branches from main
- Write tests for new behavior; keep coverage reasonable
- Follow style and CI checks; keep commits small and focused
- Open a PR with a clear description and checklist

Good first contributions:
- Add tests for edge-case TOST scenarios
- Improve docs for configuration and artifacts
- Expand validation datasets and cases

## Roadmap
- CLI for running experiments and validations
- Additional equivalence tests beyond TOST
- Built-in report generation (HTML/Markdown)
- Dataset loaders and synthetic data generators
- Cloud-friendly storage for artifacts (S3, GCS, Azure)
- Example notebooks and end-to-end templates

## License
TBD. If absent, treat as All Rights Reserved until a license is added.
