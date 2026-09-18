# Contribution roadmap

## Good first contributions

- **Dataset provenance:** locate and document the original source, version, collection
  context, and reuse terms; verify the included CSV against the source.
- **Reproducibility:** run the documented workflow on Windows/Linux/Python 3.11,
  attach environment and metrics, and document installation problems.
- **Documentation:** improve explanations and add screenshots from the maintained
  notebook, clearly distinguishing them from historical images.

## Model quality

- Add subgroup counts and uncertainty-aware metrics by age range, sex, smoking,
  and region. Avoid conclusions based on very small groups.
- Evaluate probability calibration using training-only validation and proper scores.
- Add validated SHAP/LIME explanations for the saved preprocessing pipeline, with
  checks for feature names, class ordering, and supported library versions.
- Design nested/repeated evaluation and a genuinely unseen external test dataset.
- Lock and verify an environment for all seven learned models and notebook execution.

## Product extensions after the evaluation is credible

- Add a Streamlit demonstration with input validation and clear educational wording.
- Add API/container deployment with integration tests, rather than claiming that
  the research notebook is already a deployed service.
- Build real plan recommendations only after obtaining appropriately sourced plan
  attributes, user preferences, and evaluation outcomes.

Open an insight issue to discuss scope before a large contribution. These are
proposed tasks, not claims that the features already exist.
