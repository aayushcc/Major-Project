# Step 1: Add modules to provide access to specific libraries and functions
import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci

# Step 4: Define Sumo configuration
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_CONFIG = r"C:\Users\Dell\OneDrive\Desktop\SUMO\YYY.sumocfg"

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--start",
    "--quit-on-end",
    "--step-length", "0.1"
]

# -------------------------
# Step 5: Define CNN-DQN Model and Components
# -------------------------


class DQN(nn.Module):
    def __init__(self, input_size=13, output_size=2):
        super(DQN, self).__init__()
        # 1D CNN layers for tabular data (treated as 1D signal)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        # Adjust based on conv output size
        self.fc1 = nn.Linear(32 * input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Add channel dim for Conv1d: (batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Experience Replay Buffer


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Step 6: Define Variables and Hyperparameters
# -------------------------


# State variables (same as original)
q_DR2_0 = 0
q_DR2_1 = 0
q_DR2_2 = 0
q_LD1_0 = 0
q_LD1_1 = 0
q_LD1_2 = 0
q_RU1_0 = 0
q_RU1_1 = 0
q_RU1_2 = 0
q_UL2_0 = 0
q_UL2_1 = 0
q_UL2_2 = 0
current_phase = 0

# RL Hyperparameters (adapted from original)
TOTAL_STEPS = 10000
ALPHA = 0.001  # Learning rate for optimizer
GAMMA = 0.9
EPSILON_START = 1.0  # Exploration starts high
EPSILON_END = 0.01
EPSILON_DECAY = 0.995  # Decay rate
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Steps to update target network
MAX_QUEUE = 100  # For normalization

ACTIONS = [0, 1]
MIN_GREEN_STEPS = 0
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions (Adapted for DQN)
# -------------------------


def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)


def get_state():
    # Same as original, but return as numpy array for NN
    detector_LD1_0 = "LD1_0"
    detector_LD1_1 = "LD1_1"
    detector_LD1_2 = "LD1_2"
    detector_UL2_0 = "UL2_0"
    detector_UL2_1 = "UL2_1"
    detector_UL2_2 = "UL2_2"
    detector_RU1_0 = "RU1_0"
    detector_RU1_1 = "RU1_1"
    detector_RU1_2 = "RU1_2"
    detector_DR2_0 = "DR2_0"
    detector_DR2_1 = "DR2_1"
    detector_DR2_2 = "DR2_2"
    traffic_light_id = "J1"

    q_LD1_0 = get_queue_length(detector_LD1_0)
    q_LD1_1 = get_queue_length(detector_LD1_1)
    q_LD1_2 = get_queue_length(detector_LD1_2)
    q_UL2_0 = get_queue_length(detector_UL2_0)
    q_UL2_1 = get_queue_length(detector_UL2_1)
    q_UL2_2 = get_queue_length(detector_UL2_2)
    q_RU1_0 = get_queue_length(detector_RU1_0)
    q_RU1_1 = get_queue_length(detector_RU1_1)
    q_RU1_2 = get_queue_length(detector_RU1_2)
    q_DR2_0 = get_queue_length(detector_DR2_0)
    q_DR2_1 = get_queue_length(detector_DR2_1)
    q_DR2_2 = get_queue_length(detector_DR2_2)
    current_phase = get_current_phase(traffic_light_id)

    state = np.array([q_LD1_0, q_LD1_1, q_LD1_2, q_UL2_0, q_UL2_1, q_UL2_2,
                      q_RU1_0, q_RU1_1, q_RU1_2, q_DR2_0, q_DR2_1, q_DR2_2, current_phase], dtype=np.float32)
    # Normalize queues (phase is 0-3, so leave as is or one-hot if needed)
    state[:-1] /= MAX_QUEUE
    return state


def apply_action(action, tls_id="J1"):
    global last_switch_step, current_simulation_step
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step


def get_action_from_policy(state, epsilon, policy_net):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()


def update_dqn(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        batch_size)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Current Q values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    # Loss and update
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Simulation and Comparison Function
# -------------------------


def run_simulation(use_dqn=False):
    # Initialize TraCI
    traci.start(sumo_cmd)
    print(f"TraCI connected for {'DQN' if use_dqn else 'Fixed Timing'}!")

    # Initialize DQN components if using DQN
    if use_dqn:
        policy_net = DQN()
        target_net = DQN()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        epsilon = EPSILON_START
    else:
        epsilon = 0.0  # Fixed timing

    # Reset globals
    global last_switch_step, current_simulation_step
    last_switch_step = -MIN_GREEN_STEPS
    cumulative_reward = 0.0
    step_history = []
    reward_history = []
    queue_history = []
    avg_queue_history = []

    for step in range(TOTAL_STEPS):
        current_simulation_step = step
        state = get_state()
        action = get_action_from_policy(
            state, epsilon, policy_net if use_dqn else None)
        apply_action(action)

        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward
        done = step == TOTAL_STEPS - 1  # End of episode

        if use_dqn:
            replay_buffer.push(state, action, reward, new_state, done)
            update_dqn(policy_net, target_net, optimizer,
                       replay_buffer, BATCH_SIZE, GAMMA)
            if step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if step % 100 == 0:
            total_queue = sum(new_state[:-1]) * \
                MAX_QUEUE  # Denormalize for display
            avg_queue = total_queue / 12
            print(f"Step {step}, Reward: {reward:.2f}, Cum Reward: {cumulative_reward:.2f}, Total Queue: {total_queue}, Avg Queue: {avg_queue:.2f}, Epsilon: {epsilon:.3f}")
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(total_queue)
            avg_queue_history.append(avg_queue)

    traci.close()
    if use_dqn:
        torch.save(policy_net.state_dict(), 'dqn_model.pth')
        print("DQN model saved.")
    return step_history, reward_history, queue_history, avg_queue_history


# Run Fixed Timing
print("\n=== Running Fixed Timing Simulation ===")
fixed_steps, fixed_rewards, fixed_queues, fixed_avg_queues = run_simulation(
    use_dqn=False)

# Run CNN-DQN
print("\n=== Running CNN-DQN Simulation ===")
dqn_steps, dqn_rewards, dqn_queues, dqn_avg_queues = run_simulation(
    use_dqn=True)

# -------------------------
# Step 9: Visualization and Comparison
# -------------------------

# Plot Cumulative Reward Comparison
plt.figure(figsize=(10, 6))
plt.plot(fixed_steps, fixed_rewards, marker='o',
         linestyle='-', label="Fixed Timing")
plt.plot(dqn_steps, dqn_rewards, marker='s', linestyle='-', label="CNN-DQN")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Comparison: Cumulative Reward (Fixed Timing vs CNN-DQN)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length Comparison
plt.figure(figsize=(10, 6))
plt.plot(fixed_steps, fixed_queues, marker='o',
         linestyle='-', label="Fixed Timing")
plt.plot(dqn_steps, dqn_queues, marker='s', linestyle='-', label="CNN-DQN")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Comparison: Total Queue Length (Fixed Timing vs CNN-DQN)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Average Queue Length Comparison
plt.figure(figsize=(10, 6))
plt.plot(fixed_steps, fixed_avg_queues, marker='o',
         linestyle='-', label="Fixed Timing")
plt.plot(dqn_steps, dqn_avg_queues, marker='s', linestyle='-', label="CNN-DQN")
plt.xlabel("Simulation Step")
plt.ylabel("Average Queue Length per Lane")
plt.title("Comparison: Average Queue Length (Fixed Timing vs CNN-DQN)")
plt.legend()
plt.grid(True)
plt.show()

print("\nComparison Complete. Check plots for insights. DQN may show better performance after training.")
