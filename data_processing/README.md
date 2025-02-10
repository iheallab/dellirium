# Data processing

This directory contains the code for processing the data after it has been downloaded and preprocessed. The data is processed into a format that can be used for training and evaluating the models.

- 1_get_seq.py: Obtain the time series data from all cohorts.
- 2_get_static.py: Obtain the static data from all cohorts.
- 3_get_outcomes.py: Obtain the delirium labels from all cohorts.
- 4_standardize_vars.py: Standardize some laboratory values to have the same units.
- 5_split_cohort.py: Split the cohorts into training, validation, and test sets.
- 6_add_mortality.py: Add mortality information to outcomes.