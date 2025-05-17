import datetime
import gymnasium

import numpy as np
from collections import deque

import random
import torch
from torch import nn
import yaml
import flappy_bird_gymnasium
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from networkx.generators.random_graphs import newman_watts_strogatz_graph
from rich.markup import render
from sympy import false
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import argparse

import os


#for printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"
#directory for saving runs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'agg': used to generate plots as images and saves them to a file instead of rendering to screen


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class DQN_agent():
    #agent uses device made in main
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        #hyperparameters
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a'] #learning rate
        self.discount_factor_g = hyperparameters['discount_factor_g'] #discount rate
        self.network_sync_rate = hyperparameters['network_sync_rate']  #number of steps agent takes before syncing policy and target dqn
        self.replay_memory_size = hyperparameters['replay_memory_size'] #size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size'] #size of training data set from replay memory
        self.epsilon_init = hyperparameters['epsilon_init'] #1 = 100% random action
        self.epsilon_decay = hyperparameters['epsilon_decay'] #epsilon decay
        self.epsilon_min = hyperparameters['epsilon_min'] #minimum epsilon value
        self.stop_on_reward = hyperparameters['stop_on_reward'] #stop training after reaching this number of rewards
        self.fc1_nodes = hyperparameters['fc1_nodes'] #how many nodes
        self.env_make_params = hyperparameters.get('env_make_params', {}) #get optional environment specific parameters

        self.device = device
       # self.action_set = action_set
       # self.is_training = True
      #  self.replay_memory = deque(maxlen=1000000)

        #Neural network
        self.loss_fn = nn.MSELoss() #NN loss fuction MSE=Mean Squared Error can be swapped
        self.optimizer = None #NN optimizer, initialize later

        #path to run
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=True):
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        # agent = DQN_agent(device, action_set=env.action_space)
        num_states = env.observation_space.shape[0] # expecting type: box
        num_actions = env.action_space.n #number of possible actionss
        rewards_per_episode = [] #rewards per episode
        last_graph_update_time = datetime.datetime.now()


        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device) #creating policy network

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)

            #create target network and make identical to policy
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a) #policy network optimizer

            epsilon_history = [] #list to keep track of epsilon decay

            step_counter = 0 #track number of steps taken. Used for syncing policy and target network

            best_reward = -9999999 #track best reward
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE)) #load learned policy
            policy_dqn.eval() #sswitch model to evaluation mode

        #train indefinitely, manually stop the run when satisfied with results
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            print("episode = ", episode)
            terminated = False
            episode_reward = 0.0

            #performs actions until episode terminates or reaches max reward
            while (not terminated and episode_reward < self.stop_on_reward):

                # select action based on epsilon-greedy
                if is_training and random.random() < epsilon:
                    #select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    #select best action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                #get reward
                episode_reward += reward

                #convert new state and reward to tensors on device
                new_state = torch.tensor(state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated)) #save experience into memory

                    step_counter += 1 #increment step counter

                # move to new state
                state = new_state
            #keep track of rewards per episode
            rewards_per_episode.append(episode_reward)

            #save model when new best reward is obtained
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)*100:.1f}%)"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                #update graph every x seconds
                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                    print('saving graph to ', self.GRAPH_FILE)

                #if enough experience collected
                if len(memory) > self.mini_batch_size:
                    # sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    #copy policy network to target network after number of steps
                    if step_counter > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_counter = 0

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
    def save_graph(self, rewards_per_episode, epsilon_history):
        #save plots
        fig = plt.figure(1)

        #plot average rewards (Y-axis) vs episodes (X-axis)
        mean_reward = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_reward)):
            mean_reward[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid
        plt.ylabel("Mean Reward")
        plt.plot(mean_reward)

        #plot epsilon decay (Yaxis) vs episodes (Xaxis)
        plt.subplot(122)
        plt.ylabel("Episode Decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        #save plot
        fig.savefig(self.GRAPH_FILE)
        plt.close()

    #optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #transport the list of experiences and seperate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        #stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            #calculate target Q values
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        #calculate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1)) ==>
                .squeeze()                         ==>
        '''
        #compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)
        #optimize model
        self.optimizer.zero_grad() #clear gradients
        loss.backward() #compute gradient
        self.optimizer.step() #update network parameters

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Parse command line inputs
        parser = argparse.ArgumentParser(description='Train or test model')
        parser.add_argument('hyperparameters', help='')
        parser.add_argument('--train', help='Training mode', action='store_true')
        args = parser.parse_args()

        hyperparams = args.hyperparameters
        is_training = args.train
    else:
        # Default values for running without command-line args
        hyperparams = "cartpole1"
        is_training = True

    try:
        dql = DQN_agent(hyperparameter_set=hyperparams)
        dql.run(is_training=is_training)

    except KeyboardInterrupt:
        print('[INFO] Bye bye bye')
        sys.exit(0)

