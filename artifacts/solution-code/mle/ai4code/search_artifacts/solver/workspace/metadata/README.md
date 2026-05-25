# Train/Val Split

We use an 80:20 train/validation split.
The split is performed using GroupShuffleSplit on `ancestor_id` to prevent data leakage from forked notebooks.
Random state is fixed to 42.
