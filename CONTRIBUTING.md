# Contributing

Thank you for helping improve this educational project. Code, documentation,
methodology reviews, and reproducibility reports are all welcome.

## Share insights without writing code

Open an [insight or improvement issue](https://github.com/hrnareshabd/Health-Insurance-Recommendation-System/issues/new?template=insight.md).
Describe the finding, evidence, and suggested change. Dataset provenance,
evaluation design, explainability, and clearer explanations are especially useful.
Use the bug template for a failing command or unexpected result. Do not include
private health information, credentials, or personal records.

## Make a change

1. Read the [README](README.md), [methodology](docs/METHODOLOGY.md), and [roadmap](docs/ROADMAP.md).
2. Fork the repository and clone your fork. Create a branch for one focused change.
3. Use Python 3.10–3.12 and a virtual environment. Install `requirements.txt`;
   use `requirements-notebook.txt` for notebook work and `requirements-all-models.txt`
   for the optional seven-model comparison.
4. Make the change in `insurance_model/` where possible so the notebook and command
   line share the same implementation. Keep notebook outputs cleared before committing.
5. Run `python -m unittest discover -s tests -v`. For training changes, also run
   `python -m insurance_model.pipeline train` and explain metric differences.
6. Open a pull request to `main` with the problem, change, and validation evidence.

Model changes must keep preprocessing inside the cross-validation pipeline and
must not use test scores to select a model. Report dataset checksum, seed, threshold,
versions, and split sizes with any new result. Do not treat invented profile labels
as ground truth or claim that this dataset validates insurance plan recommendations.

Please keep discussion respectful, give specific feedback, and credit others' work.
The existing MIT license applies to project code; dataset provenance and reuse terms
are tracked separately in [DATA.md](docs/DATA.md).

Maintainer: Naresh Hosahalli Rudresh — **hrnaresh39@gmail.com**.
