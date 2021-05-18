import agent

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, conv_length = 1000, fileName = None):
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.asarray(history))
    if len(history) > conv_length:
        plt.plot(np.convolve(np.asarray(history), np.ones(conv_length), 'valid') / conv_length)
    plt.show()

def stats(result):
    return (result.min(), result.max(), result.mean(), result.std())

def plot_results(results, fileName = None):
    plt.clf()
    plt.hist(results)
    if fileName:
        plt.savefig(fileName)
    plt.show()

agent = agent.DQNAgent('CartPole-v1', update_freq = 5, tau = 0.5)
agent.learn(1000)
results = agent.test(1000)
print(stats(results))
# plot_results(results)
