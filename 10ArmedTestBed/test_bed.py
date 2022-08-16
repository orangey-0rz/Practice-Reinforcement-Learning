import numpy as np
import pandas as pd

from numpy.random import default_rng

# Creatge RNG generator operator
rng = default_rng()

# Create 2000 10-armed bandits and export to CSV
pd.DataFrame(data=rng.normal(size = (2000,10))).to_csv('./10ArmedTestBed/test_bed.csv', index=False)