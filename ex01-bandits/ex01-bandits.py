import numpy as np
import matplotlib.pyplot as plt
import random
import math

class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    # iterate over range of bandit arms
    for a in possible_arms:
        # every arm is played once
        n_plays[a] = 1  # n_plays[a] += 1 is possible as well
        # get reward by playing arm
        rewards[a] = bandit.play_arm(a)
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]

    # Main loop
    while bandit.total_played < timesteps:
        # TODO: instead do greedy action selection
        # TODO: update the variables (rewards, n_plays, Q) for the selected arm
        a = np.argmax(Q)  # assume there is only one 'best' action because of random selection
        # sum up reward for played arm
        rewards[a] += bandit.play_arm(a)
        # increase amount of arm plays by one
        n_plays[a] += 1
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]


def epsilon_greedy(bandit, timesteps):
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)
    # init epsilon
    eps = 0.1

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    # iterate over range of bandit arms
    for a in possible_arms:
        # every arm is played once
        n_plays[a] = 1  # n_plays[a] += 1 is possible as well
        # get reward by playing arm
        rewards[a] = bandit.play_arm(a)
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]

    # Main loop
    while bandit.total_played < timesteps:
        var_random = np.random.uniform(0, 1)
        # probability to take random action: eps
        if var_random <= eps:
            # do random action
            a = random.choice(possible_arms)

        else:
            a = np.argmax(Q)  # assume there is only one 'best' action because of random selection

        # sum up reward for played arm
        rewards[a] += bandit.play_arm(a)
        # increase amount of arm plays by one
        n_plays[a] += 1
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]

def UCB1(bandit, timesteps):
    # possible improved methods:
    #   -   decaying epsilon
    #   -   optimized initialization
    #   -   ubc

    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    U = np.zeros(bandit.n_arms)
    T = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    # iterate over range of bandit arms
    for a in possible_arms:
        # every arm is played once
        n_plays[a] = 1  # n_plays[a] += 1 is possible as well
        # get reward by playing arm
        rewards[a] = bandit.play_arm(a)
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]
        U[a] = 0
        T[a] = Q[a] + U[a]

    # Main loop
    while bandit.total_played < timesteps:
        a = np.argmax(T)  # assume there is only one 'best' action because of random selection

        # sum up reward for played arm
        rewards[a] += bandit.play_arm(a)
        # increase amount of arm plays by one
        n_plays[a] += 1
        # compute sample-average action-value estimates
        Q[a] = rewards[a] / n_plays[a]
        U[a] = math.sqrt(2*math.log(timesteps)/n_plays[a])
        T[a] = Q[a] + U[a]

def main():
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)
    rewards_ucb1 = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        UCB1(b, n_timesteps)
        rewards_ucb1 += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    rewards_ucb1 /= n_episodes

    mean_u = np.mean(rewards_ucb1.reshape((100,10)))
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(
        np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(
        np.sum(rewards_egreedy)))
    plt.plot(rewards_ucb1, label="ucb1")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(
        np.sum(rewards_ucb1)))

    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.png')
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()