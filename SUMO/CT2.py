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

time_spent_in_current_phase_in_steps= 0
traffic_light_id = "J1"
G_MIN = 5
current_phase_index = 0

# for throughput and waiting time metrics
vehicles_passed = 0
max_waiting_time = 0 

for step in range(TOTAL_STEPS):

    traci.simulationStep()
    time_spent_in_current_phase_in_steps += 1
    state = get_state()

    J_LD = q_LD1_1 + q_LD1_2
    J_UL = q_UL2_1 + q_UL2_2   
    J_RU = q_RU1_1 + q_RU1_2
    J_DR = q_DR2_1 + q_DR2_2

    demands = [J_UL, J_RU, J_DR, J_LD]
    best = max(demands)
    if demands[current_phase_index] == 0 and ((time_spent_in_current_phase_in_steps/step_length) >= G_MIN):
        switch = True
    elif ((time_spent_in_current_phase_in_steps/step_length) >= G_MIN) and best>=(demands[current_phase_index]*1.3):
        switch = True
    else:
        switch = False
    
    if switch:
        traci.trafficlight.setPhase(traffic_light_id, (current_phase_index + 1) % 4)
        current_phase_index = (current_phase_index + 1) % 4
        time_spent_in_current_phase_in_steps = 0

         # Throughput: count vehicles that left the network this step
    vehicles_passed += len(traci.simulation.getArrivedIDList())

        # Track maximum waiting time
    for veh_id in traci.vehicle.getIDList():
        wt = traci.vehicle.getWaitingTime(veh_id)
        if wt > max_waiting_time:
            max_waiting_time = wt

    if step % 100 == 0:
        print(f"Step {step}, State: {state}")

traci.close()

# --- Final metrics ---
print(f"\nTotal vehicles passed (throughput): {vehicles_passed}")

print(f"Maximum waiting time: {max_waiting_time:.2f} seconds")

print("Simulation completed.")
