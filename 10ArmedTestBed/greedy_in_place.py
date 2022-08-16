'''
Greedy algorithm, where action is always selected by choosing maximal action value estimate

bandit(a) = normal(q(a)) = reward from choosing the action

q(a) is the actual action value

'''
from cProfile import label
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


# Create an rng generator
rng = default_rng()

# Import test bed as dataframe
test_bed = pd.read_csv('./10ArmedTestBed/test_bed.csv')

# Initialize Constants
time_steps = 1000
time = range(time_steps)
number_of_bandits = len(test_bed.index)
number_of_arms = len(test_bed.columns)
loop_count = 0

#print(number_of_bandits)
# wait = input("Press Enter to continue.")

# Get plots ready to populate
fig, axs = plt.subplots(2)

# Epsilon greedy algorithm - set epsilon and the probablility distribution for choosing action
for epsilon in [0, 0.1, 0.01]:
    loop_count = loop_count + 1

    epsilon_action_choice = [1-epsilon] + [
        epsilon/number_of_arms for i in range(number_of_arms)]
    print(epsilon_action_choice)
    #wait = input("Press Enter to continue.")

    # Rewards record to be used for graphing later
    rewards = np.zeros((number_of_bandits, time_steps))

    # We also want to graph the percentage of the time the optimal action is chosen across all bandits
    # Since this is a stationary problem, we can keep count of whether the optimal action was chosen
    # optimal_actions = np.zeros(number_of_bandits)
    optimal_actions_counter = np.zeros((number_of_bandits, time_steps))

    for ten_armed_bandit in test_bed.index:        # Returns object of type RangeIndex (pandas space saving)
        # Pull ten-armed bandit:
        bandit = test_bed.iloc[ten_armed_bandit]   # iloc for index-based locating of rows,cols

        # Track number of times a specific action has been taken
        choice_counter = np.zeros((len(bandit)))

        # Record the optimal action's index:
        optimal_idx = rng.choice(bandit[bandit[:] == max(bandit)].index.to_list())
        #print(optimal_idx)

        # Initialize:
        action_val_estimates = pd.DataFrame(data=(0 for i in range(10)))
        
        # Loop for time steps:
        for i in time:
            # Make list of indices with max values
            # print(action_val_estimates)

            max_idx = action_val_estimates[
                action_val_estimates.iloc[:,0] == max(
                    action_val_estimates.iloc[:,0])].index.to_list()
            #print(max_idx)

            # Choose index at random as the greedy action
            greedy_action = random.choice(max_idx)

            # choose any of the available actions with probability epsilon
            action = rng.choice(
                [greedy_action] + [i for i in range(number_of_arms)],
                p=epsilon_action_choice)
            #wait = input("Press Enter to continue.")

            # actions[ten_armed_bandit, i] = action

            # Update vars
            choice_counter[action] = choice_counter[action] + 1
            current_action_est = action_val_estimates.iloc[action]
            reward = rng.normal(loc=test_bed.iloc[ten_armed_bandit, action])
            #print(reward)
            
            if 0==1:
                print(type(optimal_idx))
                print(type(action))
                wait = input("Press Enter to continue.")


            # Record if optimal action was chosen
            if int(action) == int(optimal_idx):
                optimal_actions_counter[ten_armed_bandit, i] = 1

            # Test for reward array
            rewards[ten_armed_bandit, i] = reward

            # Update estimates
            action_val_estimates.iloc[action] = current_action_est + (
                reward - current_action_est)/(choice_counter[action])

    # Want to store into a 2000 x 1000 table all 1000 time steps
    rewards_frame = pd.DataFrame(data=rewards)
    optimal_actions_frame = pd.DataFrame(data=optimal_actions_counter)
    print(rewards_frame)
    print(optimal_actions_frame)

    axs[0].set_title('average reward')
    axs[1].set_title('percentage of optimal action chosen')

    axs[0].plot(time, [
        rewards_frame[i].mean() for i in time], label='epsilon={}'.format(epsilon))
    axs[0].legend(loc='lower right')
    
    axs[1].plot(time, [
        optimal_actions_frame[i].mean() for i in time], label='epsilon={}'.format(epsilon))
    axs[1].legend(loc='lower right')

    plt.savefig("full_greedy_plots.png", format='png')

    optimal_actions_frame.to_csv(
        './10ArmedTestBed/optimal_actions_frame{}.csv'.format(loop_count), index=False)

plt.close()