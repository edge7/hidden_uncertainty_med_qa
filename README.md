# Hidden-uncertainty Assessment via Non-verbalized Signatures for Medical QA

Reproducibility scripts for the paper "Hidden-uncertainty Assessment via Non-verbalized Signatures for Medical QA".

## Installation

```bash
# Clone the repository
git clone https://github.com/edge7/hidden-uncertainty-med-qa
cd hidden-uncertainty-med-qa

# Install with uv (recommended)
uv sync

# Or with pip
pip install .

# For downloading data from Zenodo, install with download extras
pip install '.[download]'
```

## Download Data

Download the experimental data from Zenodo (DOI: [10.5281/zenodo.18138856](https://doi.org/10.5281/zenodo.18138856)):

```bash
python scripts/reproduce_tables.py --download --data-dir ./data
```

This downloads ~10GB (compressed) including:
- `results/` - Per-model experimental results
- `results_common/` - Cross-model transfer experiments

## Reproduce Paper Tables

```bash
# Reproduce all tables
python scripts/reproduce_tables.py --data-dir ./data

# Reproduce a specific table
python scripts/reproduce_tables.py --data-dir ./data --table 3
python scripts/reproduce_tables.py --data-dir ./data --table 5
python scripts/reproduce_tables.py --data-dir ./data --table fig4
```

### Available Tables

| Argument | Description |
|----------|-------------|
| `--table 3` | ROC-AUC (unweighted macro average) |
| `--table 4` | Acc-Cov AUC (unweighted macro average) |
| `--table 5` | Accuracy @ Coverage levels (10%-50%) |
| `--table 7` | Universal vs Model-specific features |
| `--table 8` | Error rate by disagreement level |
| `--table fig4` | Cross-Model Transfer matrix |

## Requirements

- Python >= 3.10
- numpy >= 1.26.0, < 2.0.0 (pinned for reproducible sorting)
- scikit-learn == 1.7.2 (pinned for pickle compatibility)
- pandas >= 2.0.0
- tabulate >= 0.9.0

## Models Analyzed

- DeepSeek-R1-Distill-Qwen-32B
- Qwen3-32B
- Olmo-3-32B-Think (AllenAI)
- gpt-oss-120b
- gpt-oss-120b-high

## Datasets

MedQA, MedXpertQA-R, MedXpertQA-U, AfriMedQA, MedBullets, MedExQA, MedMCQA, MMLU-Pro, MMLU (medical), PubMedQA

## Author

Enrico D'Urso

## Citation

```bibtex
@article{hidden-uncertainty-med-qa,
  title={Hidden-uncertainty Assessment via Non-verbalized Signatures for Medical QA},
  author={D'Urso, Enrico},
  year={2026}
}
```

## License

Apache 2.0
