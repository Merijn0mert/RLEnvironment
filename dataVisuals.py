import datetime

import yaml
import networkx as nx
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for matplotlib
from matplotlib import pyplot as plt
import os
from pathlib import Path
import torch
import numpy as np

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)  # Create directory if it doesn't exist
#plt.ion()

class DataVisuals():
    def __init__(self, hyperparameter_set):
        super().__init__()
        self.hyperparameter_set = hyperparameter_set
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def saveToLog(self, best_reward, episode, mean_reward, policy_dqn):
        log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Episode {episode}: New best reward {mean_reward:0.1f} ({(mean_reward - best_reward) * 100:.1f}%)"
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')
        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
        # best_reward = mean_reward
        return best_reward

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        fig.set_size_inches(16, 10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.07, right=0.97, top=0.93, bottom=0.07)
        mean_reward = np.zeros(len(rewards_per_episode))
        # Calculate moving average of rewards
        for x in range(len(mean_reward)):
            mean_reward[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        # Graph 1 Left Top
        plt.subplot(221)
        plt.ylabel("Mean Reward")
        plt.plot(mean_reward)
        # Graph 2 Right Top
        plt.subplot(222)
        plt.ylabel("Episode Decay")
        plt.plot(epsilon_history)
        # Graph 3 full bottom (212) or left bottom (223)
        plt.subplot(212)
        plt.ylabel("Reward per episode")
        plt.plot(rewards_per_episode)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)  # Save the figure
        plt.close()

    def live_network_flow(self, model, fig_ax=None):
        plt.ion()
        device = next(model.parameters()).device
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig, ax = fig_ax
            ax.clear()

        G = nx.DiGraph()

        input_dim = model.fc1.in_features
        input_tensor = torch.randn(1, input_dim).to(device)
        input_tensor.requires_grad = True

        activations = {}
        gradients = {}

        # Forward hook to save activations
        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output
                if output.requires_grad:
                    output.register_hook(lambda grad: gradients.setdefault(name, grad))

            return hook

        # Register hooks for layers of interest
        hooks = []
        for name, module in model.named_modules():
            # For simplicity, only hook Linear layers or ReLUs
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_forward_hook(forward_hook(name)))

        model.eval()
        output = model(input_tensor)
        model.train()

        # Backward to get gradients
        output.sum().backward()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Build graph with activations keys
        positions = {}
        x_spacing = 3.0
        y_step = 1.5

        layer_names = list(activations.keys())
        for i, layer_name in enumerate(layer_names):
            act = activations[layer_name]
            for j in range(act.shape[1]):  # assuming batch size 1
                node_name = f"{layer_name}_{j}"
                G.add_node(node_name)
                positions[node_name] = (i * x_spacing, -j * y_step)

                if i > 0:
                    prev_layer = layer_names[i - 1]
                    for k in range(activations[prev_layer].shape[1]):
                        prev_node = f"{prev_layer}_{k}"

                        grad_tensor = gradients.get(layer_name, None)
                        if grad_tensor is not None:
                            grad_mag = grad_tensor.abs().mean().item()
                        else:
                            grad_mag = 0.0

                        color = (1.0 - grad_mag, 0.0, grad_mag)
                        G.add_edge(prev_node, node_name, color=color, weight=grad_mag)

        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx(G, pos=positions, ax=ax, node_size=300, edge_color=edge_colors)
        ax.set_title("Forward Flow + Gradient Paths (Live)")
        ax.axis('off')
        plt.pause(0.001)

        return fig, ax

    def plot_dqn_connections(self, model, mode='weights', max_outputs=30):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # === Left subplot: Input to Hidden (fc1) ===
        ax = axes[0]
        if mode == 'weights':
            weights = model.fc1.weight.data.cpu().numpy()
        elif mode == 'grads':
            weights = model.fc1.weight.grad.cpu().numpy()
        else:
            raise ValueError("mode must be 'weights' or 'grads'")

        num_outputs, num_inputs = weights.shape
        if num_outputs > max_outputs:
            indices = np.linspace(0, num_outputs - 1, max_outputs).astype(int)
            weights = weights[indices, :]
            num_outputs = max_outputs

        max_abs = np.abs(weights).max()
        for i in range(num_inputs):
            for j in range(num_outputs):
                w = weights[j, i]
                color = (1 - w / max_abs, 0, w / max_abs) if w >= 0 else (1, 1 + w / max_abs, 1 + w / max_abs)
                ax.plot([0, 1], [i, j], color=color, linewidth=1)

        input_labels = [f"feature_{i}" for i in range(num_inputs)]
        output_labels = [f"h_{j}" for j in range(num_outputs)]

        for i, label in enumerate(input_labels):
            ax.text(-0.05, i, label, ha='right', va='center', fontsize=8)

        for j, label in enumerate(output_labels):
            ax.text(1.05, j, label, ha='left', va='center', fontsize=6)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1, max(num_inputs, num_outputs))
        ax.set_title(f"Input → Hidden Layer ({mode.title()})")
        ax.axis('off')

        # === Right subplot: Hidden to Output ===
        ax = axes[1]

        if model.enable_dueling_dqn:
            # Get weights or grads from value and advantages
            if mode == 'weights':
                value_weights = model.value.weight.data.cpu().numpy()
                adv_weights = model.advantages.weight.data.cpu().numpy()
            else:
                value_weights = model.value.weight.grad.cpu().numpy()
                adv_weights = model.advantages.weight.grad.cpu().numpy()

            weights = np.vstack([value_weights, adv_weights])
            num_outputs, num_inputs = weights.shape
            output_labels = ['Value'] + [f'Adv_{i}' for i in range(num_outputs - 1)]

        else:
            if mode == 'weights':
                weights = model.output.weight.data.cpu().numpy()
            else:
                weights = model.output.weight.grad.cpu().numpy()

            num_outputs, num_inputs = weights.shape
            output_labels = [f'Output_{i}' for i in range(num_outputs)]

        if num_outputs > max_outputs:
            indices = np.linspace(0, num_outputs - 1, max_outputs).astype(int)
            weights = weights[indices, :]
            output_labels = [output_labels[i] for i in indices]
            num_outputs = max_outputs

        max_abs = np.abs(weights).max()
        for i in range(num_inputs):
            for j in range(num_outputs):
                w = weights[j, i]
                color = (1 - w / max_abs, 0, w / max_abs) if w >= 0 else (1, 1 + w / max_abs, 1 + w / max_abs)
                ax.plot([0, 1], [i, j], color=color, linewidth=1)

        input_labels = [f"h_{i}" for i in range(num_inputs)]

        for i, label in enumerate(input_labels):
            ax.text(-0.05, i, label, ha='right', va='center', fontsize=8)

        for j, label in enumerate(output_labels):
            ax.text(1.05, j, label, ha='left', va='center', fontsize=6)

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1, max(num_inputs, num_outputs))
        ax.set_title(f"Hidden → Output Layer ({mode.title()})")
        ax.axis('off')

        plt.tight_layout()
        plt.pause(0.001)
        return fig, axes[0]


