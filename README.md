# Health Insurance Recommendation System

**An educational machine-learning project for classifying insurance charges.**

[![Tests](https://github.com/hrnareshabd/Health-Insurance-Recommendation-System/actions/workflows/tests.yml/badge.svg)](https://github.com/hrnareshabd/Health-Insurance-Recommendation-System/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hrnareshabd/Health-Insurance-Recommendation-System/blob/main/notebooks/health_insurance_analysis.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project explores whether six demographic and lifestyle features can predict
charges above or below a training-data threshold. It includes an exploratory notebook,
reusable training and prediction code, model comparison, saved evaluation metadata,
and automated regression tests.

The repository keeps its original name, but the current dataset supports **charge
classification**, not selection of real insurance plans. This is a portfolio/research
prototype, not a validated insurance or medical decision tool.

## Start here

- **Try it:** [open the maintained notebook in Colab](https://colab.research.google.com/github/hrnareshabd/Health-Insurance-Recommendation-System/blob/main/notebooks/health_insurance_analysis.ipynb), run the setup cell, then **Runtime → Run all**. The setup downloads this repository and installs dependencies. Save a copy to Drive to edit it.
- **Understand it:** read the [methodology and limitations](docs/METHODOLOGY.md) and [dataset notes](docs/DATA.md).
- **Share insights:** [open an improvement issue](https://github.com/hrnareshabd/Health-Insurance-Recommendation-System/issues/new?template=insight.md).
- **Contribute:** read [CONTRIBUTING.md](CONTRIBUTING.md) and choose a task from the [roadmap](docs/ROADMAP.md).

## Workflow

```text
Validate CSV → remove exact duplicates → split train/test
                                         ↓
                         Training-only charge threshold
                                         ↓
                  Cross-validation of preprocessing + models
                                         ↓
                        Select by training CV F1
                                         ↓
                  Evaluate winner on held-out test rows
                                         ↓
                   Save pipeline + threshold + metrics
```

The default run compares a dummy baseline, logistic regression, and random forest.
The optional extended run compares all seven learned classifiers from the original
project (including CatBoost, XGBoost, LightGBM, SVM, and KNN), plus the dummy baseline.
The notebook adds training-data exploration and permutation feature importance.

## Run locally

Use Python **3.10–3.12**. Run commands from the repository root.

```bash
git clone https://github.com/hrnareshabd/Health-Insurance-Recommendation-System.git
cd Health-Insurance-Recommendation-System
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m insurance_model.pipeline train
python -m insurance_model.pipeline predict --profile examples/profile.json
```

On Windows PowerShell, replace the activation command with
`.venv\Scripts\Activate.ps1`. If your system uses `python3`, use that to create the
environment. Training writes `artifacts/model.joblib` and `artifacts/metrics.json`.
The example profile is fictional and has no observed outcome; it demonstrates input/output.

For notebooks:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/health_insurance_analysis.ipynb
```

For the extended model comparison:

```bash
python -m pip install -r requirements-all-models.txt
python -m insurance_model.pipeline train --all-models --output artifacts/all-models
```

If LightGBM reports a missing OpenMP library on macOS, install its platform runtime
as described in the [official LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html),
or use Colab/the default two-model run. Missing optional models fail explicitly.

## Data and prediction inputs

`Insurance.csv` contains **1,338 rows**, with **1,337 distinct records** after removing
one exact duplicate. No missing values were found in the included file.

| Field | Accepted input |
| --- | --- |
| `age` | Integer, 18–64 (dataset coverage) |
| `sex` | `female` or `male` (dataset categories) |
| `bmi` | Positive finite number; out-of-training-range values are flagged |
| `children` | Integer, 0–5 (dataset coverage) |
| `smoker` | `yes` or `no` |
| `region` | `northeast`, `northwest`, `southeast`, or `southwest` |

`charges` is used only during training. A charge above the **training partition's
median** is High Charges; a charge at or below it is Low Charges. This differs from
the original full-dataset median of 9,382.033. The exact training threshold is saved
with each model. Probabilities are uncalibrated, and no plan recommendation is returned.

## Results and reproducibility

See [verified run details](docs/RESULTS.md). Each training run saves the selected model,
CV scores, test metrics, dataset checksum, seed, and software versions in `metrics.json`.
Dependency ranges are installation constraints, not a fully locked environment.

The old 94.78% CatBoost accuracy and six-profile “83.3% recommendation accuracy” are
**historical, not verified results of this corrected workflow**. The original analysis
used full-data preprocessing and test-based selection, and the fictional profiles
did not have observed outcomes. Read the [methodology](docs/METHODOLOGY.md) before
comparing old and new scores.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the core tests and command-line train/predict workflow on Python
3.10, 3.11, and 3.12. The optional model suite and live Colab service are separate checks.

## Repository guide

```text
insurance_model/                  Shared training and prediction implementation
notebooks/health_insurance_analysis.ipynb  Maintained notebook; local + Colab setup
notebooks/archive/                Original research, marked historical
Insurance.csv                     Original dataset, preserved unchanged
examples/profile.json             Example prediction input
tests/                            Regression checks
docs/                             Data, methodology, results, contribution roadmap
.github/                          CI, issue templates, pull-request template
Result_Images/                    Historical screenshots
requirements*.txt                 Core, notebook, optional model dependencies
CONTRIBUTING.md                    How to give feedback and contribute
SECURITY.md                        Private vulnerability-reporting contact
LICENSE                           MIT license for project code
```

The old root-level notebook URL remains as a pointer to the maintained and archived versions.

## Original research

Developed by **Naresh Hosahalli Rudresh** in Google Colab.

- [Original shared Colab notebook](https://colab.research.google.com/drive/1Fk20VF42ExHkTi-eTslUbDb3J1uUfG9B?usp=sharing)
- [Archived original notebook](notebooks/archive/original_research.ipynb)
- [Historical result images](Result_Images/)

The original Colab cells were compared with the GitHub notebook during this review
and matched. The Drive notebook remains the historical original; the maintained
workflow is the GitHub notebook linked above.

## Author and contact

**Naresh Hosahalli Rudresh** · MSc Data Science · Germany

[GitHub](https://github.com/hrnareshabd) · [LinkedIn](https://www.linkedin.com/in/naresh-h-r/)
· **hrnaresh39@gmail.com**

For project questions and suggestions, prefer GitHub Issues so others can join the
discussion. Code is licensed under [MIT](LICENSE); the dataset's original source and
reuse terms still need confirmation, as tracked in [DATA.md](docs/DATA.md).
