# Train/Val Split

This directory contains the fixed train and validation splits for the competition.
- `train_ids.csv`: IDs of the training set.
- `val_ids.csv`: IDs of the validation set.

Rules followed:
- Random state: 42
- Ratio: 80% train, 20% validation
- Strategy: Stratified sampling based on target label `MGMT_value` to ensure class balance across splits.
- Filtered cases: Excluded 00109, 00123, 00709 due to known issues (per competition data description).
- ID column: `BraTS21ID`, formatted as a 5-digit string with leading zeros.
