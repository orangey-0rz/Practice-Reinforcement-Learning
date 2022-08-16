import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

# Initialize Variables
time_steps = 1000

# Create an rng generator
rng = default_rng()

# Import test bed as dataframe
test_bed = pd.read_csv('./10ArmedTestBed/test_bed.csv')

# Rewards record to be used for graphing later
rewards = np.zeros((len(test_bed.columns), time_steps))
rewards_frame = pd.DataFrame(data=rewards)
print(rewards_frame)