import os
import sys
import traci
import time

# Ensure SUMO_HOME is in the environment variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

print("SUMO_HOME environment variable found.")

# Path to your SUMO configuration file
sumocfg_file = "../../scenarios/cologne3/cologne3.sumocfg"

try:
    print("Attempting to start SUMO simulation...")
    traci.start(["sumo", "-c", sumocfg_file])
    print("SUMO simulation started successfully.")

    # Run for a few steps to allow vehicles to spawn
    for step in range(10):
        print(f"Simulation step {step}")
        traci.simulationStep()
        vehicle_count = traci.vehicle.getIDCount()
        print(f"Number of vehicles in the simulation: {vehicle_count}")

        if vehicle_count == 0 and step == 5:
            print("No vehicles spawned after 5 steps. Attempting to add a vehicle...")
            try:
                # Attempt to add a vehicle
                # Note: You need to replace 'route_id' with an actual route ID from your simulation
                traci.vehicle.add("test_vehicle", "route_id")
                print("Vehicle added successfully.")
            except traci.exceptions.TraCIException as e:
                print(f"Failed to add vehicle: {e}")

        if vehicle_count > 0:
            print("Vehicles have spawned.")
            break

        time.sleep(1)  # Wait for 1 second between steps

    if vehicle_count == 0:
        print("WARNING: No vehicles spawned after 10 simulation steps.")

    print("Attempting to retrieve simulation time...")
    sim_time = traci.simulation.getTime()
    print(f"Current simulation time: {sim_time}")

    print("Attempting to close TraCI connection...")
    traci.close()
    print("TraCI connection closed successfully.")

except traci.exceptions.FatalTraCIError as e:
    print(f"Fatal TraCI error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("Script execution completed.")