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
    #set gamma to value between 0 and 1
    #the bigger gamma the longer the convergence take
    gamma = 0.9
    
    #set delta to check tolerance
    delta = 1e-7
    #set counter
    counter = 0
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    # while loop goes as long as tolerance is not reached
    while delta > theta:
        counter += 1 #count number of iterations until convergence
        v = V_states.copy() #store 'old' state values
        #iterate over states
        for state in range(n_states):
            #initalize helper variable to identify greedy action for each state
            V_s_max = 0
            #iterate over actions 
            for action in range(n_actions):
                #initalize value function dependent on state
                V_s = 0
                #iterate over possible next states
                for i in range(len(env.P[state][action])):
                        #store probability for state transition
                        p = env.P[state][action][i][0]
                        #store the next state
                        next_state = env.P[state][action][i][1]
                        #check if next state if not terminal
                        if not env.P[state][action][i][3]:
                            #store the reward
                            r = env.P[state][action][i][2]
                        #if terminal state has value 15 it is the goal, so the reward is 1
                        elif state == 15:
                            r = 1
                        #else it is a hole so the reward is 0
                        else: 
                            r = 0
                        #compute the sum over all value functions for each state, action pair
                        V_s = V_s + (r + gamma * p * V_states[next_state])
                # only take the best value function (best action)
                if V_s > V_s_max:
                    V_s_max = V_s
            #update value function array
            V_states[state] = V_s_max
        #update delta 
        delta = max(0, np.linalg.norm(v - V_states))
        #make sure that loop is able to terminate
        if counter == 10000:
            break
    
    return 0, V_states, counter
    
                


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy, value_fcn_opt, convergence_steps = value_iteration()
    
    print("Optimal value function:")
    print(value_fcn_opt)
    print("Amount of steps for convergence")
    print(convergence_steps)
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
