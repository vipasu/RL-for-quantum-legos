#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from lego_env import Legoenv, Biased_Legoenv
import gym

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nmax", type=int)
parser.add_argument("--ntensor", type=int)
parser.add_argument("--timesteps", type=int)
parser.add_argument("--desc")
args = parser.parse_args()

def is_valid(state, action, num_max_actions=17, max_tensors=6, num_min_actions=0):
    state = np.array(state)
    env = Legoenv(max_tensors=max_tensors)
    out_state = np.zeros_like(state)
    terminal_state = np.zeros_like(state)
    terminal_state[-1] = 1
    if state[-1] == 1:
        return False, terminal_state

    # make sure not too many actions are taken
    if np.sum(state) >= num_max_actions:
        if action == env.action_space.n-1:
            out_state = state.copy()
            out_state[-1] = 1
            return True, out_state
        else:
            return False, terminal_state
    action_valid = True
    num_tensor_actions = env.num_tensor_types * env.max_tensors
    num_active_tensors = (state == 0).argmax(axis=0) # locates first 0 in the tensor arena
    num_legs_per_tensor = 5 # this will change later based on the environmnet
    num_active_legs = num_active_tensors  * num_legs_per_tensor
    if action < num_tensor_actions:
        # double check that tensor is not already selected
        # double check that previous tensor spots are already filled
        tensor_idx, tensor_type = action // env.num_tensor_types, action % env.num_tensor_types
        if state[tensor_idx] != 0 or (tensor_idx != 0 and state[tensor_idx -1] == 0):
            action_valid = False
            out_state = terminal_state
        else:
            out_state = state.copy()
            out_state[tensor_idx] = 1 + tensor_type
    elif action < num_tensor_actions + env.num_leg_combinations:
        # leg contractions
        # check that there are no collisions
        (leg1, leg2), possible_conflicted_contractions = env.get_leg_indices_from_contraction_index(action - num_tensor_actions)
        for contraction_idx in possible_conflicted_contractions:
            if state[contraction_idx + env.max_tensors] != 0:
                action_valid = False
                out_state = terminal_state
        # check that the legs in question are spawned by tensors
        if leg1 >= num_active_legs or leg2 >= num_active_legs:
            action_valid = False
            out_state = terminal_state
        if leg1//env.tensor_size == leg2 //env.tensor_size:
            action_valid = False
            out_state = terminal_state


        # check that it doesn't leave us with too few legs
        num_contractions = np.sum(state[num_tensor_actions:])
        num_contracted_legs = 2 * num_contractions
        if num_active_legs - num_contracted_legs - 2 <= env.min_legs:
            action_valid = False
            out_state = terminal_state

        else:
            out_state = state.copy()
            out_state[action] = 1
    else:
        # terminating
        if np.sum(state) < num_min_actions:
            action_valid = False
            out_state = terminal_state
        else:
            out_state = state.copy()
            out_state[-1] = 1

    return action_valid, out_state

def mask_fn(env: gym.Env) -> np.ndarray:
    s = env.state
    mask = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        if is_valid(s, a, num_max_actions=args.nmax, max_tensors=args.ntensor)[0]:
            mask[a] = True
    return np.array(mask)



def main():

    env = ActionMasker(Biased_Legoenv(max_tensors=args.ntensor), mask_fn)
    model = MaskablePPO("MlpPolicy", env, verbose=1, gamma=1,ent_coef=.01, tensorboard_log="./lego_tensorboard/")

    checkpoint_callback = CheckpointCallback(save_freq=2048, save_path="./logs/nmax{}nt{}_normerr_{}/checkpoints".format(args.nmax, args.ntensor, args.desc))
    model.learn(total_timesteps=args.timesteps,
                tb_log_name="nmax {} ntensor {} (normalized) {}".format(args.nmax, args.ntensor, args.desc),
                reset_num_timesteps=False,
                callback=checkpoint_callback)

if __name__ == '__main__':
    main()
