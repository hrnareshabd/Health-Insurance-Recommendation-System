# Methodology and limitations

## Current task

The model classifies observed charges above or below a threshold. It does not
estimate an exact premium, establish eligibility, or rank actual insurance plans.
The six predictors are age, sex, BMI, children, smoker, and region. `charges` is
used only to derive the target and is excluded from model inputs.

## Evaluation protocol

1. Validate the bundled CSV and remove exact duplicate records.
2. Split rows into 80% training and 20% test using random state 42, before computing
   the charge threshold or fitting preprocessing. This initial split is not stratified
   because labels depend on a threshold that has not yet been estimated.
3. Compute the median charge from the training partition. A charge strictly above
   that value is class 1 (High Charges); otherwise it is class 0 (Low Charges).
4. Tune candidate pipelines with shuffled three-fold stratified CV on the training
   partition. Numerical imputation/scaling and categorical one-hot encoding are fit
   inside each fold. Select by mean CV F1, with the first candidate winning a tie.
5. Evaluate only the selected, refitted pipeline on the test partition. Report
   accuracy, F1, ROC-AUC, and a confusion matrix (rows=true, columns=predicted;
   class order Low, High). Save threshold, environment, and dataset checksum.

CV scores are tuning/selection scores, not an unbiased final estimate. The target
threshold is fixed from the full outer training partition, not re-estimated per CV
fold; scores are conditional on that target definition. Nested evaluation with a
separately defined threshold is future work. Repeated development on the same test
split also weakens its independence; external validation remains necessary.

The default comparison includes a most-frequent dummy baseline, logistic regression,
and random forest. `--all-models` adds CatBoost, XGBoost, LightGBM, SVM, and KNN:
seven learned classifiers plus the dummy baseline. Missing libraries fail visibly.

## Why the original results are historical

The [original notebook](../notebooks/archive/original_research.ipynb) is retained
for provenance, with a warning at its top. It has full-dataset preprocessing,
test-based model selection, fabricated user/item groups, and collaborative scores
that incorporate test labels. Its knowledge rules compare standardized values with
raw age/BMI thresholds. Some failed dependencies could silently use a dummy model,
and summary code combines maxima belonging to different models.

The previously reported CatBoost accuracy of 94.78% and F1 of 0.9453 therefore are
not validated results of the corrected workflow. The six handcrafted example
profiles have assumed labels, not observed charge outcomes, so their 83.3% agreement
must not be presented as recommendation accuracy. Existing screenshots are historical.

## Interpretation

- Probabilities are uncalibrated model outputs, not guaranteed confidence levels.
- A high-charge prediction does not imply a plan is recommended or medically needed.
- The small dataset does not establish suitability for Germany or another market.
- Sex, age, smoking, and region may encode sensitive differences; subgroup evaluation
  and a documented feature-use rationale are needed before any real-world use.
- There are no insurer products, coverage terms, preferences, or plan-choice outcomes.
  Genuine plan recommendations need additional data and a separate evaluation task.
- Current age support is 18–64 and children 0–5. Missing/invalid categories fail
  explicitly; BMI outside the training range produces an extrapolation warning.
- The maintained notebook provides model-agnostic permutation importance on training
  data for exploration only. Historical SHAP/LIME work is in the archive; validated
  integration with the new pipeline remains on the roadmap.
