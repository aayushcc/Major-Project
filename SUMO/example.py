

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import time


os.environ["SUMO_HOME"] = r"D:\SUMO\SUMO program"
############################################
# 1. SUMO PATH SETUP
############################################

if "SUMO_HOME" not in os.environ:
    sys.exit("Please set SUMO_HOME")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

SUMO_BINARY = r"D:\SUMO\SUMO program\bin\sumo-gui.exe"
SUMO_CONFIG = r"D:\SUMO\SUMO\YYY.sumocfg"

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--start",
    "--step-length", "0.1",
    "--delay", "500"
]

############################################
# 2. PYTORCH DQN MODEL
############################################


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

############################################
# 3. HYPERPARAMETERS
############################################


STATE_SIZE = 13      # 12 detectors + 1 phase
ACTION_SIZE = 2      # 0 = keep, 1 = switch
LR = 0.001
GAMMA = 0.95
EPSILON = 0.1
TOTAL_STEPS = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(STATE_SIZE, ACTION_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

############################################
# 4. ENVIRONMENT FUNCTIONS
############################################

TLS_ID = "J1"

DETECTORS = [
    "LD1_0", "LD1_1", "LD1_2",
    "UL2_0", "UL2_1", "UL2_2",
    "RU1_0", "RU1_1", "RU1_2",
    "DR2_0", "DR2_1", "DR2_2"
]


def get_state():
    queues = [traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS]
    phase = traci.trafficlight.getPhase(TLS_ID)
    return np.array(queues + [phase], dtype=np.float32)


def get_reward(state):
    return -np.sum(state[:-1])   # minimize queues


def select_action(state):
    if random.random() < EPSILON:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        s = torch.tensor(state).unsqueeze(0).to(device)
        return torch.argmax(model(s)).item()


def apply_action(action):
    if action == 1:
        current = traci.trafficlight.getPhase(TLS_ID)
        program = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
        traci.trafficlight.setPhase(
            TLS_ID, (current + 1) % len(program.phases))

############################################
# 5. TRAINING STEP
############################################


def train_step(state, action, reward, next_state):
    s = torch.tensor(state).unsqueeze(0).to(device)
    ns = torch.tensor(next_state).unsqueeze(0).to(device)

    q_values = model(s)
    next_q = torch.max(model(ns)).detach()

    target = q_values.clone()
    target[0][action] = reward + GAMMA * next_q

    loss = loss_fn(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

############################################
# 6. MAIN LOOP
############################################


print("Starting SUMO + DQN...")
traci.start(sumo_cmd)

for step in range(TOTAL_STEPS):
    state = get_state()
    action = select_action(state)
    apply_action(action)

    traci.simulationStep()
    next_state = get_state()
    reward = get_reward(next_state)

    train_step(state, action, reward, next_state)

    if step % 100 == 0:
        print(f"Step {step} | Reward {reward:.2f}")

traci.close()
print("Training finished")
