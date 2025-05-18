import argparse
import torch.optim as optim
import numpy as np
import torch
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
import random
from collections import Counter

import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical

from VectorHaSH.grid_2d_wrapper import Grid2dWrapper
from VectorHaSH.grid_2d_recurrence_wrapper import Grid2dRecurWrapper
from VectorHaSH.grid_mlp_wrapper import GridNew2dWrapper
from VectorHaSH.grid_mlp_recurrence_wrapper import GridNew2dRecurWrapper
from VectorHaSH.grid_cells.assoc_utils_np_2D import path_integration_Wgg_2d, configurated_path_integration_Wgg_2d
from VectorHaSH.grid_cells.assoc_utils_np import pseudotrain_3d_iterative_step

from towertask.env import TowerTaskEnv
from towertask.model import RNNPolicy, select_action, finish_episode, finish_MLP_episode, SimpleMLP # leaky RNN
from towertask.utils import calculate_ema, split_tensor_by_lambdas, \
                reshape_to_square_tensors, calculate_moving_average, \
                save_args, set_seed, plot_success_rates_over_time, get_class_weights, build_model_path
from towertask.config import DATA_DIR, FIGURE_DIR, get_figure_path

from test import test_episode
                
def main(args):
    # ----------------------------
    # Setup paths and parameters
    # ----------------------------
    Np = args.Np if args.Np else min(400, np.prod(args.lambdas))
    print('Using Np =', Np)

    path, _, _ = build_model_path(args, root_dir=FIGURE_DIR)
    env_save_path, _, _ = build_model_path(args, root_dir=DATA_DIR)

    os.makedirs(path,     exist_ok=True)
    os.makedirs(env_save_path,   exist_ok=True)
    print(f"Saving model outputs to: {path}")
    print(f"Saving env data to: {env_save_path}")

    # Continue with training setup...
    best_success_rate, patience_counter = 0, 0
    checkpoints_list = []  # Initialize empty list to keep track of checkpoint paths
    
    if os.path.exists(path):
        # Refresh the entire directory
        shutil.rmtree(path)
        print(f'Deleted {path}')

    # Create the directory again
    os.makedirs(path)
    print(f'Created {path}')
    save_args(args, os.path.join(path, 'args.json'))
    
    # Initialize TowerTaskEnv
    env = TowerTaskEnv(sequence_length=args.sequence_length, fov=args.fov, path='', reset_data=True, 
                       verbose=args.verbose, max_towers=args.max_towers, q=args.q, noise_level=args.noise_level)
    
    # initialize storage variables
    use_cuda = torch.cuda.is_available()
    episode_successes, running_reward = [], 0 
    
    all_episodes_data = []
    final_action_sequences = [] # actions taken at last step of each episode (list of 0, 1, or 'timeout')
    exploration_times = [] # steps taken to explore in each episode (list of ints)
    all_episode_rewards = [] # reward of each episode (list of float)
    all_runnning_rewards = [] # running reward of each episode (list of float)
    all_timeout = [] # whether each episode times out (list of 0s and 1s)
    
    # Wrap TowerTaskEnv with GridWrapper
    if args.new_model:
        Ns = 2 * args.fov
    else:
        Ns = 2 * args.fov
    
    Ng= sum([int(e**args.dimension) for e in args.lambdas])
    arg_gcpc = {'Ng': Ng, 'Np': Np, 'Ns': Ns, 'lambdas': args.lambdas}
    
    if not args.rnn_only:
        if args.gcpc == 'gp':
            input_dimension = Np+Ng 
        elif args.gcpc == 'g':
            input_dimension = Ng 
        elif args.gcpc == 'p':
            input_dimension = Np
        else:
            assert "mode not found"
        if args.with_sensory:
            input_dimension += (2 * args.fov)
        
        if args.with_mlp:
            counter = Counter() # class counter
            class_count_cbloss = [0, 0, 0]
            if args.mlp_input_type == 'p' or args.mlp_input_type == 'p_s':
                mlp_input_size = Np
            elif args.mlp_input_type == 'sensory':
                mlp_input_size = Ns
            else:
                assert 'not implemented'
            MLP = SimpleMLP(input_size=mlp_input_size, hidden_size=args.mlp_hidden_size, output_size=3).to('cuda')
            mlp_optimizer = optim.Adam(MLP.parameters(), lr=args.mlp_learning_rate)
            mlp_success_rate_window = []
            mlp_success_rate_window_for_nonzero = []

            # do condition statement for different `grid_assignment`; generate Wggs that update only evidence modules in dir {-1, 0, 1}
            # call different path integration dict according to `grid_assignment`.
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
    else:
        input_dimension = 2 * args.fov
        if args.rnn_add_pos:
            input_dimension += 1
        if args.rnn_add_evi:
            input_dimension += 1
        if args.larger_rnn_with_scalffold_size:
            args.hidden_size += (Ng + Np)
        if args.larger_rnn_with_LEC_size:
            args.hidden_size += 2 * args.fov
            
        if args.with_mlp:
            counter = Counter() # class counter
            class_count_cbloss = [0, 0, 0]
            mlp_input_size = Ns
            MLP = SimpleMLP(input_size=mlp_input_size, hidden_size=args.mlp_hidden_size, output_size=3).to('cuda')
            mlp_optimizer = optim.Adam(MLP.parameters(), lr=args.mlp_learning_rate)
            mlp_success_rate_window = []
            mlp_success_rate_window_for_nonzero = []

    policy = RNNPolicy(input_dimension, args.hidden_size, 
                        num_layers=1, output_size=3, alpha=args.alpha)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    if args.resume_from != '':
        checkpoint = torch.load(args.resume_from)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if not args.rnn_only:
        if args.new_model or args.with_mlp:
            if args.add_recurrence:
                wrapped_env = GridNew2dRecurWrapper(env, arg_gcpc, args.gcpc, args.sigma, 
                                    use_cuda, convert_target=False, 
                                    dimension=args.dimension, conv_int=args.conv_int, 
                                    grid_step_size=args.grid_step_size,
                                    grid_assignment=args.grid_assignment,
                                    debug=args.debug, args=args, new_model=args.new_model, with_mlp=args.with_mlp)
            else:
                wrapped_env = GridNew2dWrapper(env, arg_gcpc, args.gcpc, args.sigma, 
                                    use_cuda, convert_target=False, 
                                    dimension=args.dimension, conv_int=args.conv_int, 
                                    grid_step_size=args.grid_step_size,
                                    grid_assignment=args.grid_assignment,
                                    debug=args.debug, args=args, new_model=args.new_model, with_mlp=args.with_mlp)
            
        else:
            if args.add_recurrence:
                wrapped_env = Grid2dRecurWrapper(env, arg_gcpc, args.gcpc, args.sigma, 
                                    use_cuda, convert_target=False, 
                                    dimension=args.dimension, conv_int=args.conv_int, 
                                    grid_step_size=args.grid_step_size,
                                    grid_assignment=args.grid_assignment,
                                    debug=args.debug)
            else:
                wrapped_env = Grid2dWrapper(env, arg_gcpc, args.gcpc, args.sigma, 
                                    use_cuda, convert_target=False, 
                                    dimension=args.dimension, conv_int=args.conv_int, 
                                    grid_step_size=args.grid_step_size,
                                    grid_assignment=args.grid_assignment,
                                    debug=args.debug)
    
    # Training loop 
    for episode in tqdm(range(args.num_episodes)):
        # initialize storage variables
        exploration_time = 0
        if not args.rnn_only:
            if args.new_model or args.with_mlp:
                g, p, sensory, _, p_s_processed = wrapped_env.reset(episode_idx=episode)
            else:
                g, p, sensory, _ = wrapped_env.reset(episode_idx=episode)
        else:
            state, _ = env.reset()
        spatial_axis, evidence_axis = 0, 0
        episode_reward = 0
        mlp_epoch_loss = 0
        mlp_epoch_steps = 0
        
        if args.save_state_data: 
            episode_data = {'states': [], 'p':[], 'spatial_axis':[], 'evidence_axis':[], 'action':[], 'position':[]}
        
        # while the episode is not done, record `states` (grid cell activation) and `p` (place cell activation)
        current_time = 0
        prediction_to_eviV = {0:-1, 1:0, 2:1}
        eviV_to_target = {-1:0, 0:1, 1:2}
        while True:
            # feed p to MLP, get prediction among {-1, 0, 1},   
            if not args.rnn_only:
                if args.debug:
                    print('1', reshape_to_square_tensors(split_tensor_by_lambdas(g, args.lambdas, n=2), args.lambdas, 2))
                    breakpoint()
                if args.with_mlp:     
                    if args.mlp_input_type == 'p':
                        mlp_evi_probs = MLP(p)
                    elif args.mlp_input_type == 'sensory':
                        mlp_evi_probs = MLP(sensory.view(1,-1,1).cuda())
                    elif args.mlp_input_type == 'p_s':
                        mlp_evi_probs = MLP(p_s_processed)
                    else:
                        assert 'not implemented'
                    
                    max_probs, predicted_evidence_velocity = torch.max(mlp_evi_probs.data, 2)
                    class_count_cbloss[predicted_evidence_velocity.item()] += 1
                        
                    mapped_predicted_evidence_velocity = prediction_to_eviV[predicted_evidence_velocity.item()]
                    # only update w/ evidence prediction if moved in space        
                    spatial_multiplier = 1 if wrapped_env.k == 0 else spatial_axis
                    mapped_predicted_evidence_velocity *= spatial_multiplier

                    # use evi prediction to update g
                    if not args.ground_truth:
                        g = torch.tensor(Wggs_evidence[mapped_predicted_evidence_velocity], dtype=torch.float32, device='cuda') @ g.view(1, -1, 1)
                    else:
                        g = torch.tensor(Wggs_evidence[evidence_axis], dtype=torch.float32, device='cuda') @ g.view(1, -1, 1)
                    if args.debug:
                        print('predicted evidence velocity as ', mapped_predicted_evidence_velocity)
                        print('current position is', wrapped_env.current_pos)
                        print('2', reshape_to_square_tensors(split_tensor_by_lambdas(g, args.lambdas, n=2), args.lambdas, 2))
                        breakpoint()
        
                    # Use updated g to update p_g, then ONLY pass p to RNN; 
                    if args.add_recurrence:
                        _, _, Wpg, Wsp, Wps, _, _, _, _, _, theta_sp, theta_ps, mask_pg, mask_ps, _, _, p_s, Wpp, theta_pp = wrapped_env.grid_info
                    else:
                        _, _, Wpg, Wsp, Wps, _, _, _, _, _, theta_sp, theta_ps, mask_pg, mask_ps, _, _, p_s = wrapped_env.grid_info
                    p_g = Wpg @ g
                    
                    if args.new_model:
                        if args.add_recurrence:
                            p = torch.relu(p_g+p_s+Wpp@wrapped_env.prev_p)
                        else:
                            p = torch.relu(p_g+p_s)
                        if args.modified_mixture:
                            p_for_update = torch.relu(p_g)
                        else:
                            p_for_update = torch.relu(p_g+p_s)
                    else:
                        if args.add_recurrence:
                            p = torch.relu(p_g+Wpp@wrapped_env.prev_p)
                        else:
                            p = torch.relu(p_g)
                        p_for_update = torch.relu(p_g)

                    Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p_for_update[:,:,0], sensory.view(1, -1).cuda(), use_torch=True)
                    Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory.view(1, -1).cuda(), p_for_update[:,:,0], use_torch=True)
                    if args.add_recurrence and not (wrapped_env.prev_p == 0).all() and not (p == 0).all():
                        Wpp, theta_pp = pseudotrain_3d_iterative_step(Wpp, theta_pp, wrapped_env.prev_p[:,:,0], p[:,:,0], use_torch=True, max_norm=50)   

                    # update grid info: Wsp, Wps, g, theta_sp, theta_ps    
                    wrapped_env.grid_info[3], wrapped_env.grid_info[4], wrapped_env.grid_info[7], wrapped_env.grid_info[10], wrapped_env.grid_info[11] = Wsp, Wps, g, theta_sp, theta_ps
                    if args.add_recurrence:
                        wrapped_env.grid_info[17], wrapped_env.grid_info[18] = Wpp, theta_pp
                        wrapped_env.prev_p = p.clone()

                    g = g[:,:, 0]
                    if torch.all(torch.eq(p_s, 0)) and episode > 100 and len(torch.nonzero(sensory)) != 0:
                        print('episode', episode)
                        breakpoint()
            
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

                # Compute loss of MLP prediction and true label, update model parameter.
                if args.with_mlp:
                    mlp_success_rate_window.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)
                    if evidence_axis != 0:
                        mlp_success_rate_window_for_nonzero.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)

                    counter.update([mapped_predicted_evidence_velocity])
                    weights = get_class_weights(counter)
                    mlp_criterion = nn.CrossEntropyLoss(weight=weights)
                    mlp_loss = mlp_criterion(mlp_evi_probs.view(1,-1), torch.tensor([eviV_to_target[evidence_axis]], device='cuda'))
                    
                    mlp_epoch_loss += mlp_loss
                    mlp_epoch_steps += 1
                
                action, _ = select_action(policy, state)
                current_pos = wrapped_env.env.current_position
                if args.new_model or args.with_mlp:
                    next_g, next_p, next_sensory, reward, done, info, next_spatial_axis, next_evidence_axis, next_ps = wrapped_env.step(action=action)
                else:
                    next_g, next_p, next_sensory, reward, done, info, next_spatial_axis, next_evidence_axis = wrapped_env.step(action=action)
                if args.save_state_data: 
                    episode_data['g'].append(g)
                    episode_data['p'].append(p)
                    episode_data['spatial_axis'].append(spatial_axis)
                    episode_data['evidence_axis'].append(evidence_axis)
                    episode_data['action'].append(action)
                    episode_data['position'].append(current_pos)

                g = next_g # Update 
                sensory = next_sensory
                p = next_p
                evidence_axis, spatial_axis = int(next_evidence_axis), int(next_spatial_axis)
                if args.new_model:
                    p_s_processed = torch.relu(next_ps)

            else: # rnn_only
                if args.with_mlp:
                    mlp_evi_probs = MLP(state.cuda())
                    max_probs, predicted_evidence_velocity = torch.max(mlp_evi_probs.data, 2)
                    class_count_cbloss[predicted_evidence_velocity.item()] += 1
                    mapped_predicted_evidence_velocity = prediction_to_eviV[predicted_evidence_velocity.item()]
                if args.rnn_add_pos:
                    state = torch.cat((state, torch.tensor([spatial_axis], dtype=state.dtype, device=state.device)), dim=0)
                if args.rnn_add_evi:
                    state = torch.cat((state, torch.tensor([mapped_predicted_evidence_velocity], dtype=state.dtype, device=state.device)), dim=0)
                action, _ = select_action(policy, state)
                
                if args.with_mlp:
                    mlp_success_rate_window.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)
                    if evidence_axis != 0:
                        mlp_success_rate_window_for_nonzero.append(1 if mapped_predicted_evidence_velocity == evidence_axis else 0)
                    counter.update([mapped_predicted_evidence_velocity])
                    weights = get_class_weights(counter)
                    mlp_criterion = nn.CrossEntropyLoss(weight=weights)
                    mlp_loss = mlp_criterion(mlp_evi_probs.view(1,-1), torch.tensor([eviV_to_target[evidence_axis]], device='cuda'))
                    
                    mlp_epoch_loss += mlp_loss
                    mlp_epoch_steps += 1
                
                next_state, reward, done, info, next_spatial_velocity, next_evidence_velocity = env.step(action)
                state = next_state
                
                spatial_axis, evidence_axis = int(next_spatial_velocity), int(next_evidence_velocity)
                
            if done:
                final_action_sequences.append(action)
                success = info['success']
                episode_successes.append(success) 
                policy.rewards.append(reward)
                episode_reward += reward
                all_timeout.append(0)
                break
            
            # in the ongoing episode,
            else:
                exploration_time += 1
                tmp_current_position = env.current_position if args.rnn_only else wrapped_env.env.current_position
                # """reached max_try while still not in the end, timeout"""    
                if current_time == args.max_try - 1 and tmp_current_position != args.sequence_length-1:
                    #   indicate failure (`success = 0`)
                    success = 0
                    episode_successes.append(success)
                    print(f'Episode {episode}: Timeout at step {current_time}')
                    #   indicate timeout (`1`)
                    all_timeout.append(1)
                    reward = -5 
                    policy.rewards.append(reward)
                    episode_reward += reward
                    final_action_sequences.append('timeout')
                    with open(f'{path}/success.txt', 'a') as file:
                        file.write(f"{0}\n")
                    break
                # """reached end within max_try"""
                elif tmp_current_position == args.sequence_length-1:
                    policy.rewards.append(reward)
                    episode_reward += reward
                # """still progressing, while within max_try"""
                else:
                    #   track of `current_time` and rewards.
                    current_time += 1
                    policy.rewards.append(reward)
                    episode_reward += reward
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward    
        all_episode_rewards.append(episode_reward)
        all_runnning_rewards.append(running_reward)        
        exploration_times.append(exploration_time)
        if args.save_state_data: 
            all_episodes_data.append(episode_data)
            
        finish_episode(policy, optimizer)
        if (not args.rnn_only) and args.with_mlp:
            if mlp_epoch_steps > 0:
                mlp_epoch_loss /= mlp_epoch_steps
                mlp_optimizer.zero_grad()
                mlp_epoch_loss.backward()
                mlp_optimizer.step()
            mlp_success_rate = round(calculate_moving_average(mlp_success_rate_window, window_size=5000),4)
            mlp_nonzero_success_rate = round(calculate_moving_average(mlp_success_rate_window_for_nonzero, window_size=5000),4)

        # Patience
        if args.rnn_only and args.with_mlp:
            if mlp_epoch_steps > 0:
                mlp_epoch_loss /= mlp_epoch_steps
                mlp_optimizer.zero_grad()
                mlp_epoch_loss.backward()
                mlp_optimizer.step()
            mlp_success_rate = round(calculate_moving_average(mlp_success_rate_window, window_size=5000),4)
            mlp_nonzero_success_rate = round(calculate_moving_average(mlp_success_rate_window_for_nonzero, window_size=5000),4)
        current_success_rate = round(calculate_moving_average(episode_successes, window_size=5000),4)
        
        if episode % 5000 == 0:
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'exploration_times': exploration_times,
                'running_reward': running_reward,
                'episode_successes': episode_successes
            }
            if not args.rnn_only:
                env_checkpoint = {}
                if args.with_mlp:
                    checkpoint['mlp_state_dict'] = MLP.state_dict()
                env_checkpoint['wrapped_env'] = wrapped_env.grid_info
                env_checkpoint_path = os.path.join(env_save_path, f'wrapped_env.pth')
                os.makedirs(env_save_path, exist_ok=True)
                torch.save(env_checkpoint, env_checkpoint_path)
                print('saved MESH env info at', env_checkpoint_path)
        
        if current_success_rate > best_success_rate and episode > 1000:
            best_success_rate = current_success_rate
            patience_counter = 0  # Reset patience counter
            print(f"\tNew best success rate: {best_success_rate:.4f} at episode {episode}/{args.num_episodes}. Saved checkpoint...")
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'exploration_times': exploration_times,
                'running_reward': running_reward,
                'episode_successes': episode_successes
            }
            
            if not args.rnn_only:
                env_checkpoint = {}
                if args.with_mlp:
                    checkpoint['mlp_state_dict'] = MLP.state_dict()
                env_checkpoint['wrapped_env'] = wrapped_env.grid_info
                env_checkpoint_path = os.path.join(env_save_path, 'wrapped_env.pth')
                os.makedirs(env_save_path, exist_ok=True)
                torch.save(env_checkpoint, env_checkpoint_path)
                print('saved MESH env info at', env_checkpoint_path)
            
            checkpoint_path = f'{path}/checkpoint.pth'
            torch.save(checkpoint, checkpoint_path)
            print('saved model checkpoint to', checkpoint_path)
            
        elif current_success_rate <= best_success_rate:
            if episode > 15000:
                patience_counter += 1

        # Periodic logging
        if episode % args.log_interval == 0 or episode == args.num_episodes-1:
            plot_success_rates_over_time(episode_successes, args, path)

            if not args.with_mlp:
                print(f"**Episode {episode}/{args.num_episodes} [explored {exploration_time} steps], episode reward: {episode_reward}, Running reward: {running_reward:.2f}, Train success: {current_success_rate}")
            else:
                print(f"**Episode {episode}/{args.num_episodes} [explored {exploration_time} steps], episode reward: {episode_reward}, Running reward: {running_reward:.2f}, Train success: {current_success_rate}, MLP predict success: {mlp_success_rate}, MLP nonzero predict success: {mlp_nonzero_success_rate}**")

        # Early stopping check
        if patience_counter >= args.patience:
            print(f"Stopping early at episode {episode} due to no improvement in success rate (best {best_success_rate}) for {args.patience} episodes.")
            break
        
    checkpoint = {
            'episode': episode,
            'policy_state_dict': policy.state_dict(),
            'exploration_times': exploration_times,
            'running_reward': running_reward,
            'episode_successes': episode_successes,
        }
    
    if not args.rnn_only:
        if args.with_mlp:
            checkpoint['mlp_state_dict'] = MLP.state_dict()
        env_checkpoint = {'wrapped_env':wrapped_env.grid_info}
        env_checkpoint_path = os.path.join(env_save_path, 'wrapped_env.pth')
        os.makedirs(env_save_path, exist_ok=True)
        torch.save(env_checkpoint, env_checkpoint_path)
        print('saved MESH env info at', env_checkpoint_path)
        
    torch.save(checkpoint, f'{path}/checkpoint.pth')
    print('All episodes exhausted. Saved checkpoint. Exiting...')
        
    if args.save_state_data:    
        with open(f'{path}/train_states.pkl', 'wb') as file:
            pickle.dump(all_episodes_data, file)
        print('\tsaved train_states...')
        
    with open(f'{path}/final_decision.pkl', 'wb') as file:
        pickle.dump({'final_action_sequences': final_action_sequences, 
                     'exploration_times': exploration_times,
                     'all_episode_rewards':all_episode_rewards,
                     'all_running_rewards':all_runnning_rewards,
                     'all_timeout':all_timeout
                     }, file)
        # pickle.dump(final_action_sequences, file)
    print('\tsaved final decisions...')    
    
    if not args.rnn_only:
        if args.with_mlp:
            return wrapped_env, policy, episode_successes, path, MLP, env_save_path
        else:
            return wrapped_env, policy, episode_successes, path, None, env_save_path
    else:
        return env, policy, episode_successes, path, None, env_save_path


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Tower Task Training with GridWrapper')
    parser.add_argument('--noise_level', type=float, default=0, help='Noise level in stimuli, which is the std of gaussian')
    
    parser.add_argument('--rnn_only', action='store_true', help='no MESH just RNN')
    parser.add_argument('--new_model', action='store_true', help='if true, use new model')
    parser.add_argument('--with_mlp', action='store_true', help='whether using mlp to predict evidence; to use this, new_model must be true')
    
    parser.add_argument('--mlp_learning_rate', type=float, default=5e-4, help='Learning rate for the MLPoptimizer')
    parser.add_argument('--mlp_hidden_size', type=int, default=32, help='Hidden dimension of MLP')
    parser.add_argument('--mlp_input_type', type=str, default='sensory', help='sensory OR p')
    
    parser.add_argument('--num_episodes', type=int, default=100_001, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=20, help='Number of training episodes')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--sequence_length', type=int, default=20, help='Length of the sequence for TowerTaskEnv')
    parser.add_argument('--fov', type=int, default=5, help='Field of view for TowerTaskEnv')
    parser.add_argument('--max_towers', type=int, default=5)
    parser.add_argument('--policy_type', type=str, default='RNN', help='type of rnn network to use.')
    parser.add_argument('--alpha', type=float, default=0.1, help='When alpha is small, the previous hidden state has a larger influence, making the updates to the hidden state slower.')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden dimension of policy RNN')
    parser.add_argument('--patience', type=int, default=3000, help='Number of episodes to wait for improvement before halting training.')
    parser.add_argument('--indicate_maze_pos', action='store_true', help='if true, use 99 for central arm and -99 for t-arm')
    
    parser.add_argument('--lambdas', nargs='+', type=int, default=[7,8,11])
    parser.add_argument('--grid_assignment', nargs='+', type=str, default=['both','both','both'])
    parser.add_argument('--max_try', type=int, default=200, help="max number of steps per episode")
    parser.add_argument('--grid_step_size', type=int, default=1)
    parser.add_argument('--conv_int', action='store_true')
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--gcpc', type=str, default='p')
    parser.add_argument('--sigma', type=float, default=0) # gaussian smoothing
    parser.add_argument('--with_sensory', action='store_true', help='if true, concat vectorhash vector with sensory info as rnn input')
    parser.add_argument('--debug', action='store_true', help='if true, do breakpoint')
    parser.add_argument('--Np', type=int, default=None)
    
    parser.add_argument('--trial_name', type=str, default='trial_debug')
    parser.add_argument('--resume_from', type=str, default='', help='if nonempty, resume checkpoint from this directory')
    parser.add_argument('--reset_data', action='store_true', help='Use new data every time')
    parser.add_argument('--save_state_data', action='store_true', help='save state data every time')
    parser.add_argument('--verbose', action='store_true', help='print debug statement')
    parser.add_argument('--ground_truth', action='store_true', help='Use ground truth grid evidence velocity for the tower task')
    parser.add_argument('--q', type=float, default=1, help='With probability q, the maze length is sequence_length, with probability 1-q the maze length varies') 
    
    parser.add_argument('--modified_mixture', action='store_true', help='if true, feed rnn mixture p, but update Wps, Wsp using nonmixture p')
    parser.add_argument('--seed', type=int, default=42)
    
    ## rebuttal
    parser.add_argument('--rnn_add_pos', action='store_true', help='if true, concat pos velocity to sensory input for RNN only case')
    parser.add_argument('--rnn_add_evi', action='store_true', help='if true, concat evi velocity to sensory input for RNN only case')
    parser.add_argument('--larger_rnn_with_scalffold_size', action='store_true', help='if true, hidden_size += Np + Ng')
    parser.add_argument('--larger_rnn_with_LEC_size', action='store_true', help='if true, hidden_size += Ns')
    
    parser.add_argument('--add_recurrence', action='store_true', help='if true, concat evi velocity to sensory input for RNN only case')
    
    args = parser.parse_args()
    set_seed(args.seed)

    wrapped_env, trained_policy, episode_successes, path, MLP, env_save_path = main(args)
    plot_success_rates_over_time(episode_successes, args, path)
    
    if not args.rnn_only:
        os.makedirs(env_save_path, exist_ok=True)
        print('Saving .mat file to', env_save_path)
        
        OOD_test = [
            [20, 5, 'Testing'],                 # sanity check
        ]

        env = wrapped_env if args.rnn_only else wrapped_env.env
        env.q = 1 # env with fixed length

        for seq_length, max_towers, comments in OOD_test:
            env.max_towers = max_towers
            env.fixed_sequence_length = seq_length
            print(f'{comments}: {max_towers} max towers, {seq_length} seq length')
            test_episode(wrapped_env, trained_policy, args,
                        num_episodes=500,
                        path=env_save_path,
                        MLP=MLP,
                        save_files=True) 