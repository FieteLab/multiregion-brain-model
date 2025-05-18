import argparse
import torch
import torch.optim as optim
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
import pandas as pd
from scipy.io import savemat
import json
import random

from VectorHaSH.grid_cells.assoc_utils_np_2D import path_integration_Wgg_2d, configurated_path_integration_Wgg_2d
from VectorHaSH.grid_2d_wrapper import Grid2dWrapper
from VectorHaSH.grid_mlp_wrapper import GridNew2dWrapper
from VectorHaSH.grid_2d_recurrence_wrapper import Grid2dRecurWrapper
from VectorHaSH.grid_mlp_recurrence_wrapper import GridNew2dRecurWrapper
from towertask.utils import calculate_ema, split_tensor_by_lambdas, \
                            reshape_to_square_tensors, calculate_moving_average, \
                            load_args, set_seed, build_model_path
from towertask.config import DATA_DIR, FIGURE_DIR, get_figure_path
from towertask.env import TowerTaskEnv
from towertask.model import RNNPolicy, SimpleMLP, select_action, finish_episode

def test_episode(wrapped_env, policy, args, num_episodes=100, path='', MLP=None, save_files=False):
    """
    Test the policy on a set of episodes.

    Args:
    - env: The environment to test the policy on.
    - policy: The trained policy model.
    - num_episodes: Number of episodes to test on.
    - save_state_data: Boolean indicating whether to save state data.
    - path: Path to save any results or state data.

    Returns:
    - test_success_rate: The success rate over the test episodes.
    - test_rewards: List of rewards from each test episode.
    """
    policy.eval()
    wrapped_env.is_eval = True
    if not args.rnn_only: 
        Np = min(400, np.prod([7,8,11]))
        Ng= sum([int(e**2) for e in args.lambdas])
    # column_names = ['episode', 'step', 'accumulated_evidence', 'position'] + [f'neuron_{i}' for i in range(Np)]
    # df = pd.DataFrame(columns=column_names)
    
    # test_rewards = []
    test_successes = []
    steps_per_episode = []
    mlp_success_rate_window = []
    mlp_success_rate_window_for_nonzero = []
    # test_choices = []
    if save_files:
        data_info = []
        hidden = []
        ps = []
        gs = []
    if args.with_mlp and not args.rnn_only: 
        prediction_to_eviV = {0:-1, 1:0, 2:1}
        eviV_to_target = {-1:0, 0:1, 1:2}

        Wggs_evidence = {}
        for evidence_velocity in [-1, 0, 1]:
            # generate Wggs_evidence for updating evidence, with keys {-1, 0, 1}
            if args.grid_assignment == ['position','position','position']: # get Wggs that update evidence axis for all modules
                Wggs_evidence[evidence_velocity] = path_integration_Wgg_2d(args.lambdas, Ng, 
                                                                        axis=1, 
                                                                        direction=evidence_velocity)
            # generate Wggs_evidence that does not modify position-tracking module, and does modify evidence-tracking module
            else:
                module_configs = []
                for m in args.grid_assignment:
                    if m == 'position':
                        module_configs.append((0, 0))
                    elif m == 'evidence':
                        module_configs.append((1, evidence_velocity))
                    else:
                        assert False, 'unfound grid assignment'
                Wggs_evidence[evidence_velocity] = configurated_path_integration_Wgg_2d(args.lambdas, Ng, module_configs) # note that new_model parameter is false here, to update evidence submatrix
        
    for episode in tqdm(range(num_episodes)):
        if not args.rnn_only:
            if args.new_model or args.with_mlp:
                g, p, sensory, _ , p_s_processed = wrapped_env.reset(episode_idx=episode+120_000)
            else:
                g, p, sensory, _  = wrapped_env.reset(episode_idx=episode+120_000)
            evidence_axis, spatial_axis = 0, 0
        else:
            state, _ = wrapped_env.reset()

        episode_data = []
        hidden_states = []
        grid_states = []
        place_cell_states = []
        
        step = 0
        accumulated_evidence = 0
        
        done = False
        
        while True:
            if not args.rnn_only:
                if args.with_mlp:
                    if args.mlp_input_type == 'p':
                        mlp_output = MLP(p)
                    elif args.mlp_input_type == 'sensory':
                        mlp_output = MLP(sensory.view(1,-1,1).cuda())
                    elif args.mlp_input_type == 'p_s':
                        mlp_output = MLP(p_s_processed)
                    else:
                        assert 'not implemented'
                        
                    _, predicted_evidence_velocity = torch.max(mlp_output.data, 2)
                    predicted_evidence_velocity = predicted_evidence_velocity.item()
                    mapped_predicted_evidence_velocity = prediction_to_eviV[predicted_evidence_velocity]
                    spatial_multiplier = 1 if wrapped_env.k == 0 else spatial_axis
                    mapped_predicted_evidence_velocity *= spatial_multiplier
                    
                    if args.debug:
                        print('1', reshape_to_square_tensors(split_tensor_by_lambdas(g, args.lambdas, n=2), args.lambdas, 2))
                        breakpoint()
                    # Update g using predicted evidence velocity
                    g = torch.tensor(Wggs_evidence[mapped_predicted_evidence_velocity], dtype=torch.float32, device='cuda') @ g.view(1, -1, 1)
                    if args.debug:
                        print('predicted evidence velocity as ', mapped_predicted_evidence_velocity, 'while true label is', evidence_axis)
                        print('2', reshape_to_square_tensors(split_tensor_by_lambdas(g, args.lambdas, n=2), args.lambdas, 2))
                        breakpoint()
                        
                    if args.add_recurrence:
                        _, _, Wpg, Wsp, Wps, _, _, _, _, _, theta_sp, theta_ps, mask_pg, mask_ps, _, _, p_s, Wpp, _ = wrapped_env.grid_info
                    else:
                        _, _, Wpg, Wsp, Wps, _, _, _, _, _, theta_sp, theta_ps, mask_pg, mask_ps, _, _, p_s = wrapped_env.grid_info
                    p_g = Wpg @ g
                    if args.new_model: 
                        if args.add_recurrence:
                            p = torch.relu(p_g+p_s+Wpp@wrapped_env.prev_p)
                        else:
                            p = torch.relu(p_g+p_s)
                    else:
                        if args.add_recurrence:
                            p = torch.relu(p_g+Wpp@wrapped_env.prev_p)
                        else:
                            p = torch.relu(p_g)
                    s_hat = Wsp @ p
                    wrapped_env.grid_info[7] = g
                    g = g[:,:, 0]
                
                if args.with_sensory:  
                    if args.gcpc == 'g':
                        state = torch.cat((g, sensory.view(1,-1).to(g.device)),1)
                    elif args.gcpc == 'p':
                        state = torch.cat((p.view(1,-1).to(g.device), sensory.view(1,-1).to(g.device)),1)
                    else:
                        assert 'not implemented'
                else:
                    if args.gcpc == 'gp':
                        state = torch.cat((g, p.view(1,-1).to(g.device)),1)
                    elif args.gcpc == 'p':
                        state = p
                    elif args.gcpc == 'g':
                        state = g
                    else:
                        assert 'not implemented'
                if args.with_mlp:
                    mlp_success_rate_window.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)
                    if evidence_axis != 0:
                        mlp_success_rate_window_for_nonzero.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)
            
            action, hidden_state, policy_probabilities = select_action(policy, state, return_prob_dist=True)
            if args.rnn_only:
                current_pos = wrapped_env.current_position
                next_state, reward, done, info = wrapped_env.step(action)
                # get spatial axis
                if action == 2 and current_pos < wrapped_env.sequence_length - 1:
                    # if in central arm and choose move forward
                    spatial_axis = 1
                else:
                    spatial_axis = 0

                # get evidence axis
                length = len(state) // 2
                left_fov = state[:length]
                right_fov = state[length:]
                latest_obs = (int(left_fov[0]), int(right_fov[0]))
                evidence_map = {(0,1):1, (1,0):-1, (1,1):0, (0,0):0, (-1,-1):0}
                # evidence_axis = 0 if (wrapped_env.same_as_prev_pos and not wrapped_env.k == 1) else evidence_map[latest_obs] 
                evidence_axis = wrapped_env.true_local_evidence[current_pos]
                accumulated_evidence += evidence_axis 
                
                state = next_state

                if save_files:
                    episode_data.append([
                        episode,
                        step,
                        current_pos,
                        evidence_axis, # current right - left; 
                        accumulated_evidence, # cumulative right - cumulative left up to this position
                        wrapped_env.total_evidence, # total right - total left in this episode
                        action,
                        wrapped_env.info['ground_truth_action'],
                        wrapped_env.label,
                        spatial_axis,
                        policy_probabilities[0],
                        policy_probabilities[1],
                        policy_probabilities[2],
                        # Success to be added later
                    ])
                    hidden_states.append(hidden_state.detach().cpu().numpy().flatten())

            else:
                current_pos = wrapped_env.env.current_position
                if args.new_model or args.with_mlp:
                    next_g, next_p, next_sensory, reward, done, info, next_spatial_axis, next_evidence_axis, next_ps = wrapped_env.step(action=action)
                else:
                    next_g, next_p, next_sensory, reward, done, info, next_spatial_axis, next_evidence_axis = wrapped_env.step(action=action)
                accumulated_evidence += evidence_axis

                if save_files:
                    episode_data.append([
                        episode,
                        step,
                        current_pos,
                        evidence_axis, # current right - left; 
                        accumulated_evidence, # cumulative right - cumulative left up to this position
                        wrapped_env.env.total_evidence, # total right - total left in this episode
                        action,
                        wrapped_env.env.info['ground_truth_action'],
                        wrapped_env.env.label,
                        spatial_axis,
                        policy_probabilities[0],
                        policy_probabilities[1],
                        policy_probabilities[2],
                        # Success to be added later
                    ])
                
                    place_cell_states.append(p.detach().cpu().numpy().flatten())
                    grid_states.append(g.detach().cpu().numpy().flatten()) 
                    hidden_states.append(hidden_state.detach().cpu().numpy().flatten())

                g = next_g
                sensory = next_sensory
                p = next_p
                evidence_axis, spatial_axis = next_evidence_axis, next_spatial_axis
                if args.new_model:
                    p_s_processed = torch.relu(next_ps)
            
            step += 1
            
            if done:
                test_successes.append(info['success'])
                steps_per_episode.append(step)
                for data in episode_data:
                    data.append(info['success'])
                break
        
        if save_files:
            data_info.extend(episode_data)
            if not args.rnn_only:
                ps.extend(place_cell_states)
                gs.extend(grid_states)
            hidden.extend(hidden_states)
     
    if save_files:   
        info_array = np.array(data_info)
        if not args.rnn_only:
            ps_array = np.array(ps)
            gs_array = np.array(gs)
        hidden_array = np.array(hidden)
    
        # Create a dictionary of data
        if not args.rnn_only:
            data_dict = {
                'info': info_array,
                'ps': ps_array,
                'gs': gs_array,
                'hidden': hidden_array
            }
        else:
            data_dict = {
                'info': info_array,
                'hidden': hidden_array
            }
        # Save the data as .mat file
        sensory_naming = '_with_sensory' if args.with_sensory else ''
        if path != '':
            savemat(f'{path}/{num_episodes}trials.mat', data_dict)
            print(f'Data saved in .mat format to {path}/{num_episodes}trials.mat')

    test_success_rate = sum(test_successes) / num_episodes
    average_steps = np.mean(steps_per_episode)
    print(f"Test Success Rate: {test_success_rate}, Average steps taken: {average_steps}")
    if not args.rnn_only and args.with_mlp:
        print(f"MLP Success Rate: {np.mean(mlp_success_rate_window)}")
        print(f"MLP Success Rate (nonzero): {np.mean(mlp_success_rate_window_for_nonzero)}")

    return test_success_rate, steps_per_episode, average_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tower Task Testing with GridWrapper')
    parser.add_argument('--rnn_only', action='store_true', help='no MESH just RNN')
    
    parser.add_argument('--mlp_hidden_size', type=int, default=32, help='Hidden dimension of MLP')
    parser.add_argument('--mlp_input_type', type=str, default='sensory', help='sensory OR p')
    
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of the sequence for TowerTaskEnv')
    parser.add_argument('--fov', type=int, default=5, help='Field of view for TowerTaskEnv')
    parser.add_argument('--max_towers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden dimension of policy RNN')
    parser.add_argument('--alpha', type=float, default=0.1, help='When alpha is small, the previous hidden state has a larger influence, making the updates to the hidden state slower.')
    parser.add_argument('--lambdas', nargs='+', type=int, default=[7,8,11])
    parser.add_argument('--grid_assignment', nargs='+', type=str, default=['position','position','position'])
    parser.add_argument('--gcpc', type=str, default='p')
    parser.add_argument('--new_model', action='store_true', help='if true, use new model')
    parser.add_argument('--with_mlp', action='store_true', help='if true, use MLP')
    parser.add_argument('--Np', type=int, default=800)
    
    parser.add_argument('--with_sensory', action='store_true', help='if true, concat grid vector with sensory info')
    parser.add_argument('--policy_type', type=str, default='RNN', help='type of rnn network to use.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--q', type=float, default=1, help='With probability q, the maze length is sequence_length, with probability 1-q the maze length varies') 
    
    parser.add_argument('--debug', action='store_true', help='if true, print debug statements')
    parser.add_argument('--trial_name', type=str, default='trial_debug')
    parser.add_argument('--model_type', type=str, default='RNN', help='which model to use.')
    parser.add_argument('--trial', type=str, default='1', help='which trial to use.')
    
    # rebuttal
    parser.add_argument('--add_recurrence', action='store_true', help='if true, concat evi velocity to sensory input for RNN only case')
    
    default_args = parser.parse_args()
    
    # VectorHaSH setup
    if not default_args.rnn_only:
        Np = min(800, np.prod(default_args.lambdas))
        if default_args.Np:
            Np = default_args.Np
            print('Using Np =', default_args.Np)
        Ns = 2 * default_args.fov
        Ng= sum([int(e**2) for e in default_args.lambdas])
        arg_gcpc = {'Ng': Ng, 'Np': Np, 'Ns': Ns, 'lambdas': default_args.lambdas}
        
        if default_args.gcpc == 'gp':
            input_dimension = Np+Ng 
        elif default_args.gcpc == 'g':
            input_dimension = Ng 
        elif default_args.gcpc == 'p':
            input_dimension = Np
        else:
            assert "mode not found"
        if default_args.with_sensory:
            input_dimension += (2 * default_args.fov)
        if default_args.mlp_input_type == 'p' or default_args.mlp_input_type == 'p_s':
            mlp_input_size = Np
        elif default_args.mlp_input_type == 'sensory':
            mlp_input_size = Ns
        else:
            assert 'not implemented'
    else:
        input_dimension = 2 * default_args.fov

    model_path, base_parts, _ = build_model_path(default_args, root_dir=FIGURE_DIR)
    data_path, base_parts, _ = build_model_path(default_args, root_dir=DATA_DIR)
    arg_path = os.path.join(model_path, 'args.json')
    saved_args = load_args(arg_path)
    print('Loaded saved arguments from', arg_path)

    if saved_args is not None:
        # We priortize current argument's max_tower, sequence_length, and q.
        args = saved_args
        if default_args.max_towers != saved_args.max_towers:
            args.max_towers = default_args.max_towers
        print(f'Using max tower Uniform(1,{args.max_towers})')
        if default_args.sequence_length != saved_args.sequence_length:
            args.sequence_length = default_args.sequence_length
        print(f'Using environment size: {args.sequence_length}')
        args.model_type = default_args.model_type
        args.q = default_args.q
    else:
        args = default_args
        print('No saved arguments; falling to default arguments...')

    # load `policy`` checkpoint and grid_info from checkpoint
    policy = RNNPolicy(input_dimension, args.hidden_size, 
                        num_layers=1, output_size=3, alpha=args.alpha)
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint.pth'))
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    MLP = None
    if args.with_mlp:
        MLP = SimpleMLP(input_size=mlp_input_size, hidden_size=args.mlp_hidden_size, output_size=3).to('cuda')
        MLP.load_state_dict(checkpoint['mlp_state_dict'])
        MLP.eval()
    
    env = TowerTaskEnv(sequence_length=args.sequence_length, fov=args.fov, path='', 
                       reset_data=True, 
                       verbose=False, 
                       max_towers=args.max_towers,
                       q=args.q)
    if not args.rnn_only: # Load grid_info and wrap
        grid_info = torch.load(os.path.join(data_path, 'wrapped_env.pth'))['wrapped_env']
        gcpc_dict = {'Np': args.Np, 'Ns': 2*args.fov, 'Ng': sum(e**2 for e in args.lambdas), 'lambdas': args.lambdas}

        wrapper_cls = {
            (True, True): GridNew2dRecurWrapper,
            (True, False): GridNew2dWrapper,
            (False, True): Grid2dRecurWrapper,
            (False, False): Grid2dWrapper
        }[(args.new_model or args.with_mlp, args.add_recurrence)]

        wrapped_env = wrapper_cls(env, gcpc_dict, args.gcpc, sigma=0,
                                  use_cuda=True, convert_target=False, dimension=2, conv_int=False,
                                  grid_step_size=1, grid_assignment=args.grid_assignment,
                                  debug=False, grid_info=grid_info, args=args, is_eval=True,
                                  new_model=args.new_model, with_mlp=args.with_mlp)
    else:
        wrapped_env = env
    
    print('saving .mat to', data_path)
    test_success_rate, steps_per_episode, average_steps = test_episode(wrapped_env, policy, args=args, num_episodes=1000, path=data_path, MLP=MLP, save_files=True)