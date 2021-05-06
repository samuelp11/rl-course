import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    #initialize value function for all , theta and gamma
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.
    gamma = 0.99
    
    #set delta to 1
    delta = 1e-7
    #set V_s_max to 0
    V_s_max = 0 
    #set counter to 0
    counter = 0
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    
    while delta > theta:
        counter += 1
        #delta = 0
        v = V_states.copy()
        #print(v)
        for state in range(n_states):
            V_s_max = 0
            for action in range(n_actions):
                V_s = 0
                #there are either 3 possibilities to continue or 1 if it s the end
                for i in range(len(env.P[state][action])):
                    p = env.P[state][action][i][0]
                    next_state = env.P[state][action][i][1]
                    r = env.P[state][action][i][2]
                    V_s = V_s + (r + gamma * p * V_states[next_state])
                if V_s > V_s_max:
                    V_s_max = V_s
                    
            V_states[state] = V_s_max
        delta = max(0, np.linalg.norm(v - V_states))
        if counter == 1:
            break
    print(counter)
    print(v - V_states)
    print(np.linalg.norm(v - V_states))
    print(V_states)
    print(delta)
    
    return 0, V_states
    
                


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy, value_fcn_opt = value_iteration()
    
    print("Optimal value function:")
    print(value_fcn_opt)
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
