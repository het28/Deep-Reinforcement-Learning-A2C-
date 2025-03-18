import sys
import os
from datetime import datetime
from typing import List, Any, Optional, Union, Tuple
import argparse
import math
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from wager.simulation import SumoSimulation
from wager.logger import Logging
from wager.util import set_sumo_env, SimInterface
from wager.map_plot.net_parse import SumoNetwork
from wager.map_plot.render_frontend import Scene, Sequence

from model import ActorCritic, SharedAdam

SUMO_HOME = os.environ['SUMO_HOME']

class Simulator(SumoSimulation):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.waiting_time = 5
        self.n_states = 7  # Increased state size

        # Dynamically calculate the number of actions based on traffic light phases
        self.n_actions = self.determine_max_actions()

        self.waiting_times = {}
        self.lane_densities = {}
        self.queue_lengths = {}
        self.vehicle_speeds = {}
        self.total_vehicles = 0
        self.avg_travel_time = 0
        self.lane_lengths = None

    def determine_max_actions(self) -> int:
        max_actions = 0
        tlss = self.interface.trafficlight.getIDList()

        for tls in tlss:
            program_logics = self.interface.trafficlight.getAllProgramLogics(tls)
            phases = program_logics[0].phases  # Retrieve phases of the current traffic light program
            max_actions = max(max_actions, len(phases))

        print(f"Determined max actions (phases): {max_actions}")
        return max_actions

    def determine_lane_lengths(self) -> List[float]:
        tlss = self.interface.trafficlight.getIDList()
        lane_lengths = []

        for tls in tlss:
            lanes = self.interface.trafficlight.getControlledLanes(tls)
            for lane in lanes:
                lane_lengths.append(self.interface.lane.getLength(lane))
        return lane_lengths

    def calculate_lane_metrics(self, lanes):
        waiting_times = []
        lane_densities = []
        queue_lengths = []
        vehicle_speeds = []
        total_vehicles = self.interface.vehicle.getIDCount()

        if total_vehicles <= 0:
            return [0] * len(lanes), [0] * len(lanes), [0] * len(lanes), [0] * len(lanes), 0, 0

        avg_travel_time = np.mean([self.interface.vehicle.getWaitingTime(veh) for veh in self.interface.vehicle.getIDList()])

        for j, lane in enumerate(lanes):
            num_vehicles = self.interface.lane.getLastStepVehicleNumber(lane)
            lane_density = (num_vehicles / self.lane_lengths[j]) / total_vehicles
            waiting_time = self.interface.lane.getWaitingTime(lane) / total_vehicles
            queue_length = self.interface.lane.getLastStepHaltingNumber(lane)
            avg_speed = self.interface.lane.getLastStepMeanSpeed(lane)

            lane_densities.append(lane_density)
            waiting_times.append(waiting_time)
            queue_lengths.append(queue_length)
            vehicle_speeds.append(avg_speed)

        return lane_densities, waiting_times, queue_lengths, vehicle_speeds, total_vehicles, avg_travel_time

    def calculate_state(self) -> List[torch.Tensor]:
        tlss = self.interface.trafficlight.getIDList()
        states = []
        self.waiting_times = {}
        self.lane_densities = {}
        self.queue_lengths = {}
        self.vehicle_speeds = {}

        if self.lane_lengths is None:
            self.lane_lengths = self.determine_lane_lengths()

        for tls in tlss:
            lanes = self.interface.trafficlight.getControlledLanes(tls)
            lane_densities, waiting_times, queue_lengths, vehicle_speeds, total_vehicles, avg_travel_time = self.calculate_lane_metrics(lanes)

            self.lane_densities[tls] = lane_densities
            self.waiting_times[tls] = waiting_times
            self.queue_lengths[tls] = queue_lengths
            self.vehicle_speeds[tls] = vehicle_speeds
            self.total_vehicles = total_vehicles
            self.avg_travel_time = avg_travel_time

            state_features = [
                sum(lane_densities),
                sum(waiting_times),
                sum(queue_lengths),
                sum(vehicle_speeds),
                len(lanes),
                total_vehicles,
                avg_travel_time
            ]
            states.append(torch.tensor(state_features, dtype=torch.float32))

        return states

    def calculate_reward(self) -> torch.Tensor:
        """
        Calculate the reward based on vehicle speeds and other metrics.
        """
        # Flatten vehicle speeds if they are lists
        flattened_speeds = [
            speed for speeds in self.vehicle_speeds.values()
            for speed in (speeds if isinstance(speeds, list) else [speeds])
        ]

        # Speed reward (encourage faster movement)
        speed_rewards = [0.02 * sp for sp in flattened_speeds if sp]

        # Throughput reward (encourage completion of routes)
        vehicle_throughput_reward = 0.1 * self.total_vehicles

        # Aggregate reward components
        total_reward = sum(speed_rewards) + vehicle_throughput_reward

        # Log reward components for debugging
        self.reward_components = {
            "speed_rewards": sum(speed_rewards),
            "vehicle_throughput_reward": vehicle_throughput_reward,
        }

        return torch.tensor(total_reward)

    def execute_actions_on_all_tls(self, actions: torch.Tensor) -> None:
        print("Executing Actions on Traffic Lights:")
        for tls, action in zip(self.tls_datas, actions):
            phases = tls.phases
            num_phases = len(phases)

            print(f"  Traffic Light {tls.id}: {num_phases} phases available. Selected action: {action.item()}")

            if num_phases == 0:
                print(f"No phases available for traffic light {tls.id}. Skipping.")
                continue

            action_index = action.item() % num_phases
            calculated_phase = tls.deprecated_calc_new_phase(phases[action_index])
            self.interface.trafficlight.setRedYellowGreenState(tls.id, calculated_phase)
            tls.depr_last_phase = calculated_phase

        for tls in self.invalid_tls_ids:
            RYGDef = self.interface.trafficlight.getCompleteRedYellowGreenDefinition(tls)
            phase = (RYGDef[0].phases[0].state).replace('r', 'G').replace('y', 'G')
            self.interface.trafficlight.setRedYellowGreenState(tls, phase)

        for _ in range(self.waiting_time):
            self.simulation_step()
        self.emission.accumulate()


def evaluate(scenario: str, max_steps: int, log: bool = False, log_project: str = None,
             log_map: str = None) -> None:
    set_sumo_env(os.path.abspath(SUMO_HOME))
    sim = Simulator(use_nema=False,
                    sumo_config_dir=f'../../scenarios/{scenario}',
                    sumo_home_path=SUMO_HOME,
                    cfg_name=f'{scenario}.sumocfg',
                    net_name=f'{scenario}.net.xml',
                    max_steps=max_steps,
                    interface=SimInterface.TRACI)

    logger = Logging(sim,
                     project=log_project,
                     entity="otto-von-guericke-university-ailab",
                     group="Dummy",
                     name=f"{scenario}/{datetime.now().strftime('%H_%M_%S_%d_%m_%Y')}")
    logger.set_status(log)
    sim.register_logger(logger)

    model = ActorCritic(
        n_states=sim.n_states,
        n_actions=sim.n_actions
    )
    optimizer = SharedAdam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    gamma = 0.95

    sim.reset()
    tls_count = len(sim.interface.trafficlight.getIDList())
    if tls_count == 0:
        print("No traffic lights detected in the simulation.")
        return

    state, _, terminate = sim.step(torch.randint(0, sim.n_actions, (tls_count,), dtype=torch.int))
    timestep = 0
    while not terminate:
        state_tensor = torch.stack(state)
        action_probs, _ = model(state_tensor)

        # Temperature scaling for exploration-exploitation balance
        temperature = max(0.1, 1.0 - timestep / max_steps)
        scaled_probs = torch.softmax(action_probs / temperature, dim=-1)

        action_dist = torch.distributions.Categorical(probs=scaled_probs)
        actions = action_dist.sample()

        print(f"Step {timestep} Actions and Metrics:")
        for idx, action in enumerate(actions):
            tls_id = list(sim.waiting_times.keys())[idx]
            print(f"  Traffic Light {tls_id}: Action {action.item()}")
            print(f"    Action Probabilities: {scaled_probs[idx].tolist()}")
            print(f"    Waiting Time: {sim.waiting_times[tls_id]}")
            print(f"    Queue Length: {sim.queue_lengths[tls_id]}")
            print(f"    Lane Density: {sim.lane_densities[tls_id]}")
            logger.log(f"action_traffic_light_{tls_id}", action.item())
            logger.log(f"action_probabilities_traffic_light_{tls_id}", scaled_probs[idx].tolist())

        sim.execute_actions_on_all_tls(actions)
        state, rewards, terminate = sim.step(actions)

        # Compute policy loss for gradient descent with entropy regularization
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        entropy_coeff = max(0.01, 0.05 * (1 - timestep / max_steps))
        policy_loss = -(log_probs * rewards.detach()).mean() - entropy_coeff * entropy


        # Gradient ascent on rewards
        reward_gradient = rewards.sum().item()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Log metrics
        logger.log("reward", rewards.sum().item())
        logger.log("policy_loss", policy_loss.item())
        logger.log("reward_gradient", reward_gradient)
        logger.log("entropy", entropy.item())
        logger.log("mean_waiting_time", np.mean([sum(wt) for wt in sim.waiting_times.values()]) if sim.waiting_times else 0)
        logger.log("mean_lane_density", np.mean([sum(ld) for ld in sim.lane_densities.values()]) if sim.lane_densities else 0)

        print(f"Step {timestep} Metrics:")
        print(f"  Rewards: {rewards.sum().item()}")
        print(f"  Policy Loss: {policy_loss.item()}")
        print(f"  Reward Gradient: {reward_gradient}")
        print(f"  Entropy: {entropy.item()}")

        timestep += 1

    sim.reset()
    logger.finish()
    sim.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For parsing values regarding the location of the weights")
    parser.add_argument("-c", "--scenario",
                        dest="scenario",
                        help="The name of the environment",
                        default="cologne1",
                        choices=['cologne1', 'cologne3', 'cologne8', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21'])
    parser.add_argument("-ms", "--max_steps",
                        dest="max_steps",
                        help="Maximum number of steps in sumo",
                        default=1000,
                        type=int)
    parser.add_argument("-m", "--log_map",
                        dest="log_map",
                        help="Name of the .wager log map file",
                        default=None)
    parser.add_argument("-l", "--log",
                        dest="log",
                        help="Enable log to wandb.",
                        default=True)
    parser.add_argument("-lp", "--log_project",
                        dest="log_project",
                        help="Name of the wandb project",
                        default=None,
                        choices=['DLPS_DeepQLearning', 'DLPS_ActorCritics', 'DLPS_PolicyOptimisation'])
    args = parser.parse_args()

    print(args.__dict__)

    evaluate(**args.__dict__)
