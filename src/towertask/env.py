import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import random
# --------------------------
# Data Generation
# --------------------------
from towertask.data import generate_data_with_max_towers

# --------------------------
# Environment Setup
# --------------------------
"""for running with MESH, REINFORCE, 3 actions, ==> used in train.py"""
class TowerTaskEnv:
    def __init__(self, sequence_length, fov=10, 
                 reset_data=True, path='', verbose=False,
                 indicate_maze_pos=False, max_towers=None, q=1, noise_level=0):
        self.seed = 42
        self.fixed_sequence_length = sequence_length
        # self.batch_size = batch_size
        self.fov = fov
        self.path = path
        self.current_position = 0
        self.save_data = True
        self.verbose = verbose
        self.indicate_maze_pos = indicate_maze_pos
        self.fixed_max_towers = max_towers
        self.q = q
        self.noise_level = noise_level
        self.k = 0
        self.data, self.label, self.total_evidence, self.true_local_evidence = self._generate_data()
        self.prev_action = None
        self.reset_data = reset_data
        self.straight_count = 0
        self.done = False
        
    def _generate_data(self, episode_idx=0):
        # Determine sequence length based on probability q (default = 1)
        # As q increases above 0, we randomly change the environment with probablity 1-q.
        # Else the environment is fixed.
        if random.random() < self.q:
            self.sequence_length = self.fixed_sequence_length
            self.max_towers = self.fixed_max_towers
        else:
            # e.g., maze length varies from 15 to 30 if given `fixed_sequence_length=15`
            self.sequence_length = random.choice([self.fixed_sequence_length + i for i in range (-5, 11)])
            # e.g., one may also vary the max number of towers on the rewarded size
            self.max_towers = random.choice([self.sequence_length // 4 + i for i in range(-1, 2)])

        data, label, total_evidence, true_local_evidence = generate_data_with_max_towers(num_samples=1,
                                                 sequence_length=self.sequence_length, 
                                                 fov=self.fov,
                                                 max_towers=self.max_towers,
                                                 noise_level=self.noise_level, seed=self.seed + episode_idx)

        self.decision_region_start = self.sequence_length
        self.decision_region_end = self.sequence_length + 1
        
        if self.save_data and self.path != '':
            data_to_save = {'data': data, 'label': label}
            with open(f'{self.path}/dataAndLabel.pkl', 'wb') as file:
                pickle.dump(data_to_save, file)
            self.save_data = False
        return data.squeeze(0), label.item(), total_evidence, true_local_evidence

    def reset(self, episode_idx=0):
        self.current_position = 0
        if self.reset_data:
            self.data, self.label, self.total_evidence, self.true_local_evidence = self._generate_data(episode_idx)
        initial_obs = self.data[self.current_position]
        self.same_as_prev_pos = False
        return initial_obs, self.info

    def step(self, action):
        self.prev_action = action
        if self.verbose:
            print(f'At position {self.current_position} and chose {action}, while correct label is {self.label}')
        if action == 2:
            self.straight_count += 1 
        else:
            self.straight_count = 0
        

        is_turn_action = action != 2           
        # At T-arm
        if self.current_position >= self.sequence_length - 1:
            # Determine reward or penalty at the end of the data
            if is_turn_action:
                done = True
                self.current_position += 1
                self.same_as_prev_pos = False

                # We reward or penalize the turning decision made at the very end
                if action == self.label:
                    reward = 10 
                else:
                    reward = 0
            # Penalty for no decision made by the end (agent chose going forward)
            else:
                done = False
                self.same_as_prev_pos = True
                reward = -1 
        # At corridor
        else:
            done = False
            # Agent moves forward
            if action == 2:
                self.current_position += 1
                reward = 0.01
                self.same_as_prev_pos = False
            # Agent zig-zagging (L or R)
            else:
                reward = -0.001 # penalty for turning actions in the corridor before T-arm; adjust as needed
                self.same_as_prev_pos = True
        try:
            next_obs = self.data[self.current_position]
        except:
            next_obs = None
            
        # Agent moves or not determines the spatial velocity
        if action == 2 and self.current_position < self.sequence_length - 1:
            spatial_velocity = 1
        else:
            spatial_velocity = 0
        # Evidence velocity is nonzero if agent has advanced its position (observe new evidence) in the corridor (not T-arm).
        evidence_velocity = 0 if (self.same_as_prev_pos and not self.k == 1) or (self.current_position > self.sequence_length - 1) else self.true_local_evidence[self.current_position]
            
        self.done = done
        if self.verbose:
            print(f'\t Reward is {reward}, Done == {done}')
            
        self.k += 1
        return next_obs, reward, done, self.info, spatial_velocity, evidence_velocity
        
    @property
    def info(self):
        success = 0
        if self.done and self.prev_action == self.label:
            success = 1
            
        if self.current_position  >= self.sequence_length - 1:
            ground_truth_action = self.label
        else:
            ground_truth_action = 2
        
        info = {
            'current_pos': self.current_position,
            'success': success,
            'prev_action': self.prev_action,
            'ground_truth_action': ground_truth_action
        }
        return info
