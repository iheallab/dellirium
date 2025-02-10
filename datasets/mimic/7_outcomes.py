# Import libraries

import pandas as pd
import numpy as np
import math

from variables import OUTPUT_DIR

# Load brain status file

brain_status = pd.read_csv(f'{OUTPUT_DIR}/raw/brain_status.csv')

# Load ICU stays file

icustays = pd.read_csv(f'{OUTPUT_DIR}/intermediate/icustays.csv')

# Merge ICU stays and admissions

icustays = icustays.merge(brain_status, how='left', on='stay_id')

del brain_status

print(icustays['coma'].value_counts())
print(icustays['delirium'].value_counts())
print(icustays.shape)

# Save as csv file

icustays.to_csv(f'{OUTPUT_DIR}/intermediate/outcomes.csv', index=None)