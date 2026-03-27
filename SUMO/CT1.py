import os
import sys
import traci

os.environ["SUMO_HOME"] = r"D:\SUMO\SUMO program"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_BINARY = r"D:\SUMO\SUMO program\bin\sumo-gui.exe"
SUMO_CONFIG = r"D:\SUMO\SUMO\YYY.sumocfg"

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CONFIG,
    "--start",
    "--quit-on-end",
    "--step-length", "0.1",
    "--scale", "0.09"
]
step_length = 0.1 #update this value in sumo_cmd as well

traci.start(sumo_cmd)
print("TraCI connected successfully!")
traci.gui.setSchema("View #0", "real world")

# Variables for queue lengths from detectors and current phase
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

TOTAL_STEPS = 5000


def get_state():
    global q_DR2_0, q_DR2_1, q_DR2_2, q_LD1_0, q_LD1_1, q_LD1_2, q_RU1_0, q_RU1_1, q_RU1_2, q_UL2_0, q_UL2_1, q_UL2_2, current_phase

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

    q_LD1_0 = traci.lane.getLastStepHaltingNumber("LD1_0")
    q_LD1_1 = traci.lane.getLastStepHaltingNumber("LD1_1")
    q_LD1_2 = traci.lane.getLastStepHaltingNumber("LD1_2")

    q_UL2_0 = traci.lane.getLastStepHaltingNumber("UL2_0")
    q_UL2_1 = traci.lane.getLastStepHaltingNumber("UL2_1")
    q_UL2_2 = traci.lane.getLastStepHaltingNumber("UL2_2")

    q_RU1_0 = traci.lane.getLastStepHaltingNumber("RU1_0")
    q_RU1_1 = traci.lane.getLastStepHaltingNumber("RU1_1")
    q_RU1_2 = traci.lane.getLastStepHaltingNumber("RU1_2")

    q_DR2_0 = traci.lane.getLastStepHaltingNumber("DR2_0")
    q_DR2_1 = traci.lane.getLastStepHaltingNumber("DR2_1")
    q_DR2_2 = traci.lane.getLastStepHaltingNumber("DR2_2")

    current_phase = get_current_phase(traffic_light_id)

    return (q_LD1_0, q_LD1_1, q_LD1_2, q_UL2_0, q_UL2_1, q_UL2_2, q_RU1_0, q_RU1_1, q_RU1_2, q_DR2_0, q_DR2_1, q_DR2_2, current_phase)


def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

print("\n=== Starting Simulation ===")
# --- Replacement code: Throughput + Waiting/Delay Metrics ---

vehicles_passed = 0         # total vehicles that left the network
total_waiting_time = 0      # sum of waiting time of all vehicles
total_time_loss = 0         # sum of time loss of all vehicles

#for green time calculation
current_phase_index = 0
phase_time_left = 0
green_times = []
CYCLE_LENGTH = 60
G_MIN = 1

traffic_light_id = "J1"

for step in range(TOTAL_STEPS):

    traci.simulationStep()
    state = get_state()

    J_LD = q_LD1_1 + q_LD1_2
    J_UL = q_UL2_1 + q_UL2_2   
    J_RU = q_RU1_1 + q_RU1_2
    J_DR = q_DR2_1 + q_DR2_2

    demands = [J_UL, J_RU, J_DR, J_LD]
    N = len(demands)

    if phase_time_left <= 0:

        if current_phase_index == 0:
            total_demand = sum(demands)

            if total_demand == 0:
                green_times = [CYCLE_LENGTH / N] * N
            else:
                green_times = [
                    G_MIN + (CYCLE_LENGTH - N * G_MIN) * (d / total_demand)
                    for d in demands
                ]

        # set current phase
        traci.trafficlight.setPhase(traffic_light_id, current_phase_index)

        # assign its duration
        phase_time_left = max(1, int(green_times[current_phase_index] / step_length))

        # move to next phase for next switch
        current_phase_index = (current_phase_index + 1) % N

    phase_time_left -= 1

    # Throughput: count vehicles that left the network this step
    vehicles_passed += len(traci.simulation.getArrivedIDList())

    # Optional: sum waiting time and time loss for all active vehicles
    for veh_id in traci.vehicle.getIDList():
        total_waiting_time += traci.vehicle.getWaitingTime(veh_id)
        total_time_loss += traci.vehicle.getTimeLoss(veh_id)

    if step % 100 == 0:
        print(f"Step {step}, State: {state}")

traci.close()

# --- Final metrics ---
print(f"\nTotal vehicles passed (throughput): {vehicles_passed}")

if total_time_loss > 0:
    waiting_ratio = total_waiting_time / total_time_loss
    print(f"Fraction of pure waiting vs total delay: {waiting_ratio:.2f}")

print("Simulation completed.")