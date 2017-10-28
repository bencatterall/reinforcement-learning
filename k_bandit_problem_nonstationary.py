
# coding: utf-8

import operator
import numpy as np
from matplotlib import pyplot as plt

#k-bandit non-stationary problem
#
# perform a random walk over the q*(a) values
# Uses a constant step size for the values a.k.a exponential, recency-weighted average 

#Here we have k levers to pull
#Choose the true reward of action a: q*(a) using a standard normal distribution
#When an agent picks action a, return an approximate reward Q(a) by sampling from a normal distribution
#with mean q*(a) and variance 1
# greedily pick the best one, break ties by uniform probability
#have an eta: randomly explore rather than exploit with probability eta


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

def run_game(k, num_steps, eta, alpha):
    avg_values = []
    qopt = np.random.normal(0, 1, 10)
    values = {i : 0.0 for i in range(0,k)}
    for i in range(0, num_steps):
        a = pickAction(k, eta, values)
        reward = np.random.uniform(qopt[a], 1)
        values[a] = alpha*reward + (1 - alpha)*values[a]
        avg_values.append(calculate_average_game_values(values))
        qopt = [i + np.random.normal(0,0.01) for i in qopt]
    return avg_values

def run_simulation(k, eta, num_steps, num_runs, alpha):
    values = []
    for i in range(0, num_runs):
        values.append(run_game(k, num_steps, eta, alpha))
    return np.mean(values, axis=0)

def plot_results(results, etas, num_steps):
    plt.figure()
    for eta, run in zip(etas,results):
        plt.plot(range(0,num_steps), run, label = eta)
    plt.legend()
    plt.show()

def run_experiment(k, etas, num_steps, num_runs,alpha):
    results = []
    for eta in etas:
        results.append(run_simulation(k, eta, num_steps, num_runs,alpha))
    plot_results(results, etas, num_steps)

run_experiment(10, [0.0,0.01, 0.1],10000,1000, 0.1)

