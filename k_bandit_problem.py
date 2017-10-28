
# coding: utf-8


import operator
import numpy as np
from matplotlib import pyplot as plt
#k-bandit problem
#Here we have k levers to pull
#Choose the true reward of action a: q*(a) using a standard normal distribution
#When an agent picks action a, return an approximate reward Q(a) by sampling from a normal distribution
#with mean q*(a) and variance 1
# greedily pick the best one, break ties by uniform probability
#have an eta: randomly explore rather than exploit with probability eta
k = 10
eta = 0.0
num_steps = 1000

def calculate_average_game_values(values):
    sum = 0.0
    for k, v in values.items():
        sum += v
    return sum/len(values)

#Assume values are 0 -> k
def pickAction(k, eta, values):
    if eta > 0 and np.random.uniform(0, 1) < eta:
        return np.random.randint(0, k)
    
    maxElementKey = max(values, key=values.get)
    maxElements = list(filter(lambda key : values[key] == values[maxElementKey], values.keys()))
    return maxElements[np.random.randint(0,len(maxElements))]

def run_game(k, num_steps, eta):
    avg_values = []
    qopt = np.random.normal(0, 1, 10)
    action_taken = {i : 0 for i in range(0,k)}
    reward_action = {i : 0.0 for i in range(0,k)}
    values = {i : 0.0 for i in range(0,k)}
    for i in range(0, num_steps):
        a = pickAction(k, eta, values)
        reward = np.random.uniform(qopt[a], 1)
        action_taken[a] = action_taken[a] + 1
        reward_action[a] = reward_action[a] + reward
        values[a] = reward_action[a] / action_taken[a]
        avg_values.append(calculate_average_game_values(values))
    return avg_values

def run_simulation(k, eta, num_steps, num_runs):
    values = []
    for i in range(0, num_runs):
        values.append(run_game(k, num_steps, eta))
    return np.mean(values, axis=0)

def plot_results(results, etas, num_steps):
    plt.figure()
    for eta, run in zip(etas,results):
        plt.plot(range(0,num_steps), run, label = eta)
    plt.legend()
    plt.show()

def run_experiment(k, etas, num_steps, num_runs):
    results = []
    for eta in etas:
        results.append(run_simulation(k, eta, num_steps, num_runs))
    plot_results(results, etas, num_steps)

run_experiment(10, [0.0,0.01, 0.1],1000,1000)

