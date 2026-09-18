# Verified results

Verified locally on 2026-09-18 with Python 3.12.14 on macOS arm64.

| Run | Selected by CV | CV F1 | Test accuracy | Test F1 | Test ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | Random Forest | 0.9226 | 92.16% | 0.9231 | 0.9588 |
| All seven learned models + baseline | CatBoost | 0.9278 | 94.78% | 0.9478 | 0.9609 |

Both runs use 1,069 training rows and 268 test rows after deduplication, random
state 42, and training-median charge threshold **9,290.1395**. The configurations
were selected by training CV, not by these test scores. They share the same test
split, so the two runs are not independent replications. These results do not
establish performance on other datasets or real insurance plan choices.

[Core machine-readable report](results-core.json) · [Extended report](results-all-models.json)

## Validation performed

- Seven regression tests passed, including train/test separation, training-only
  threshold and scaling, invalid inputs, and saved-model prediction round-trip.
- Default and extended training completed with RuntimeWarnings treated as errors.
- The command-line prediction example completed successfully.
- All seven maintained notebook code cells executed successfully in a fresh local
  Jupyter kernel. Live Google Colab execution is not claimed; the setup cell is
  provided for contributors to run there.

## Environment notes

The exact ML package versions and dataset checksum are included in each JSON report.
For this macOS verification, LightGBM/XGBoost resolved OpenMP using the runtime
already bundled with scikit-learn, via a process-local library search path. No
system-wide installation was changed. Other macOS users may need the OpenMP
runtime described in the official LightGBM installation guide linked in the README.
The default pipeline does not need these optional boosting libraries.

NumPy is constrained below 2 and SciPy below 1.16 to avoid numerical/runtime and
solver compatibility warnings observed in this verification environment. The ranges
are not an exhaustive compatibility guarantee. Future dependency updates should
include rerunning tests and the notebook.

## Historical comparison

The original CatBoost accuracy rounds to the same 94.78%, but that does not make
it the same experiment. The new split, target threshold, preprocessing, selection
procedure, and F1/AUC differ. Keep historical screenshots separate from these reports.
