# Add modules to provide access to specific libraries and functions
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Traci allows communication between python and SUMO
import traci

#  Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'YYY.sumocfg',
]
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

Sumo_config = [
    SUMO_BINARY,
    "-c", "YYY.sumocfg"
]


#  Open connection between SUMO and Traci

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_CONFIG = r"C:\Users\Dell\OneDrive\Desktop\SUMO\YYY.sumocfg"

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--start",
    "--quit-on-end",
    "--step-length", "5",
    "--scale", "3"
]

traci.start(sumo_cmd)
print("TraCI connected successfully!")
traci.gui.setSchema("View #0", "real world")

# Variables for RL State (queue lengths from detectors and current phase)
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


# ---- Reinforcement Learning Hyperparameters ----
# The total number of simulation steps for continuous (online) training.
TOTAL_STEPS = 500

# Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
ALPHA = 0.1
# If α = 0, you ignore the new estimate and never update the Q-value.
# Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
GAMMA = 0.9
# If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
# Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random
EPSILON = 0.1

# The discrete action space (0 = keep phase, 1 = switch phase)
ACTIONS = [0, 1]


# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 200
last_switch_step = -MIN_GREEN_STEPS

#  Define Functions


def build_model(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    model = keras.Sequential()                                 # Feedforward neural network
    model.add(layers.Input(shape=(state_size,)))               # Input layer
    model.add(layers.Dense(24, activation='relu')
              )             # First hidden layer
    model.add(layers.Dense(24, activation='relu')
              )             # Second hidden layer
    model.add(layers.Dense(action_size, activation='linear'))  # Output layer
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


def to_array(state_tuple):
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))


    # Create the DQN model
state_size = 13
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)


def get_max_Q_value_of_state(s):
    state_array = to_array(s)
    Q_values = dqn_model.predict(state_array, verbose=0)[
        0]  # shape: (action_size,)
    return np.max(Q_values)


def get_reward(state):
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])
    reward = -float(total_queue)
    return reward


def get_state():
    global q_DR2_0, q_DR2_1, q_DR2_2, q_LD1_0, q_LD1_1, q_LD1_2, q_RU1_0, q_RU1_1, q_RU1_2, q_UL2_0, q_UL2_1, q_UL2_2,  current_phase

    # Detector IDs for Node1-2-EB
    detector_LD1_0 = "LD1_0"
    detector_LD1_1 = "LD1_1"
    detector_LD1_2 = "LD1_2"

    # Detector IDs for Node2-7-SB
    detector_UL2_0 = "UL2_0"
    detector_UL2_1 = "UL2_1"
    detector_UL2_2 = "UL2_2"

    # Detector IDs for Node2-3-WB
    detector_RU1_0 = "RU1_0"
    detector_RU1_1 = "RU1_1"
    detector_RU1_2 = "RU1_2"

    # Detector IDs for Node2-5-NB
    detector_DR2_0 = "DR2_0"
    detector_DR2_1 = "DR2_1"
    detector_DR2_2 = "DR2_2"

    # Traffic light ID
    traffic_light_id = "J1"

    # Get queue lengths from each detector
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

    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)

    return (q_LD1_0, q_LD1_1, q_LD1_2, q_UL2_0, q_UL2_1, q_UL2_2, q_RU1_0, q_RU1_1, q_RU1_2, q_DR2_0, q_DR2_1, q_DR2_2, current_phase)


def apply_action(action, tls_id="J1"):
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
      - Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step

    if action == 0:
        # Do nothing (keep current phase)
        return
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step = current_simulation_step


def update_Q_table(old_state, action, reward, new_state):
    """
    In DQN, we do a single-step gradient update instead of a table update.
    """
    # Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model.predict(old_state_array, verbose=0)[0]
    #  Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)

    # Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * \
        (reward + GAMMA * best_future_q - Q_values_old[action])

    # Train (fit) the DQN on this single sample
    dqn_model.fit(old_state_array, np.array([Q_values_old]), verbose=0)


def get_action_from_policy(state):
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))


def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


# Fully Online Continuous Learning Loop

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # keep this variable for apply_action usage

    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)

    traci.simulationStep()  # Advance simulation by one step

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_Q_table(state, action, reward, new_state)

    # Print Q-values for the old_state right after update
    updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]

    # Record data every 100 steps
    if step % 1 == 0:
        updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths


#  Close connection between SUMO and Traci
traci.close()

# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o',
         linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o',
         linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
