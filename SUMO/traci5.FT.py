# Step 1: Add modules to provide access to specific libraries and functions
import sys
import os
import time
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization


# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
# Static network information (such as reading and analyzing network files)
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'YYY.sumocfg',
]
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

Sumo_config = [
    SUMO_BINARY,
    "-c", "YYY.sumocfg",
    "--step-length", "0.01"
]


# Step 5: Open connection between SUMO and Traci

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_CONFIG = r"C:\Users\Dell\OneDrive\Desktop\SUMO\YYY.sumocfg"

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--start",
    "--quit-on-end",
    "--step-length", "0.01"
]

traci.start(sumo_cmd)
print("TraCI connected successfully!")
traci.gui.setSchema("View #0", "real world")


# -------------------------
# Step 6: Define Variables
# -------------------------

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
TOTAL_STEPS = 10000

# Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
ALPHA = 0.1
# If α = 0, you ignore the new estimate and never update the Q-value.
# Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
GAMMA = 0.9
# If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
# Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random
EPSILON = 0.0

# The discrete action space (0 = keep phase, 1 = switch phase)
ACTIONS = [0, 1]

# Q-table dictionary: key = state tuple, value = numpy array of Q-values for each action
Q_table = {}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 0
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------


def get_max_Q_value_of_state(s):  # 1. Objective Function 1
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])


def get_reward(state):  # 2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward


def get_state():  # 3.& 4. Constraint 3 & 4
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


def apply_action(action, tls_id="J1"):  # 5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
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
            last_switch_step = current_simulation_step


def update_Q_table(old_state, action, reward, new_state):  # 6. Constraint 6
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))

    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_table[old_state][action]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(new_state)
    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_table[old_state][action] = old_q + ALPHA * \
        (reward + GAMMA * best_future_q - old_q)


def get_action_from_policy(state):  # 7. Constraint 7
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return 0


def get_queue_length(detector_id):  # 8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):  # 8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------


# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    # action = get_action_from_policy(state)
    # apply_action(action)

    traci.simulationStep()  # Advance simulation by one step

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    # update_Q_table(state, action, reward, new_state)

    # Print Q-values for the old_state right after update
    # updated_q_vals = Q_table[state]

    # Record data every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Current_State: {state}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths
        print("Current Q-table:")
        for st, qvals in Q_table.items():
            print(f"  {st} -> {qvals}")

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# Print final Q-table info
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o',
         linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Fixed Timing: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o',
         linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Fixed Timing: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
