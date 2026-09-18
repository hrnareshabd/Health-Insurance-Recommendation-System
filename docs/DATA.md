# Dataset notes

The repository includes `Insurance.csv` with 1,338 records and seven columns:
`age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`.
Inspection found no missing values and one exact duplicate; the maintained loader
removes that duplicate before splitting, leaving 1,337 records. The source CSV is
preserved unchanged.

SHA-256 of the included file:
`6e3894cf7d9210613f62c866684055decea694345cd4f244ecd7167041578c68`.

The original README/notebook does not identify a verifiable download URL, original
publisher, collection period, or dataset license. These remain **unconfirmed**.
The file resembles a commonly used insurance-charge teaching dataset, but resemblance
alone is not provenance. Do not attribute a source or extend the code's MIT license
to third-party data without evidence.

The original analysis displayed charges as dollars; currency, collection context,
and current market applicability have not been independently verified. Region values
are dataset labels, not evidence of suitability for insurance decisions in Germany.

## A useful first contribution

Provide the original dataset source URL, version/download date if known, reuse terms,
and a checksum or row-level comparison to establish which data this repository contains.
Document any transformations. Do not upload personal or sensitive health records.

## Target definition

The maintained workflow derives the median charge from the training partition only.
It uses `charges > training_median` as High Charges and all other charges as Low Charges.
The complete-file median, 9,382.033, is historical descriptive context, not the current
training threshold. There are no real plan IDs, product details, preferences, or
user-plan interaction outcomes in this file.
