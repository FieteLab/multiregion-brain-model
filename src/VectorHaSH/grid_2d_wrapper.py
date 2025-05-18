import numpy as np
import torch
import copy
import random

from VectorHaSH.gridUtils import grid_cell_initial_2d, run_grid_cells_2d 
from VectorHaSH.base_wrapper import BaseWrapper
from towertask.utils import split_tensor_by_lambdas, reshape_to_square_tensors, set_seed

class Grid2dWrapper(BaseWrapper):
    def __init__(self, env, arg_gcpc, mode='gp', sigma=0, use_cuda=True, 
                 grid_info=None, convert_target=False, dimension=2, conv_int=True, 
                 grid_step_size=1, grid_assignment=['position', 'position', 'evidence'],
                 debug=False, is_eval=False, task_type='evidence-based'):
        super().__init__(env)
        self.task_type = task_type # evidence-based (tower), context-based, memory-based
        self.use_cuda = use_cuda
        self.current_internal_loc = None
        self.dimension = dimension
        self.grid_step_size = grid_step_size
        self.grid_assignment = grid_assignment
        self.is_eval = is_eval

        self.init_gcpc(arg_gcpc, grid_info)
        self.lambdas = arg_gcpc['lambdas']
        if grid_assignment is not None:
            assert len(grid_assignment) == len(arg_gcpc['lambdas']), "each module must have an assignment"
        
        self.sanity_mode = False
        self.verbose = False
        self.sigma = sigma
        self.mode = mode
        self.k = 0
        # self.set_force_prev = set_force_prev
        self.prev_action_in_zero = 0
        self.gps = None
        self.last_g = None
        self.allocated_grids = None
        self.convert_target = convert_target

        # self.conv_int = conv_int
        self.debug = debug

    def init_gcpc(self, arg_gcpc, grid_info):
        self.Np = arg_gcpc['Np']
        self.Ns = arg_gcpc['Ns']
        self.Ng = arg_gcpc['Ng']
        self.lambdas = arg_gcpc['lambdas']
        print('init')
        
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps = grid_cell_initial_2d(self.Np, self.Ns, self.lambdas, 
                                                                                                                                    self.use_cuda, self.dimension, 
                                                                                                                                    self.grid_assignment)
        self.g_origin = g
        torch.cuda.empty_cache()
        self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps]
        if grid_info is not None:
            # if len(grid_info) > 2:
            #     self.grid_info[1], self.grid_info[2] = grid_info[0:2]
            # else:
            #     self.grid_info[1], self.grid_info[2] = grid_info
            print('***Loaded grid info!***')
            self.grid_info = grid_info
            self.grid_info[7] = self.g_origin

    def reset(self, episode_idx=0):
        obs, info = self.env.reset(episode_idx)
        # since this reset() gives obs at position 0, we should update evidence axis by setting `same_as_prev_pos` False.
        # obs, p, _, _ = self.convert_obs(obs, prev_action=2, same_as_prev_pos=False, force_prev=self.k > 0)
        # update the current position
        self.current_pos = info['current_pos']
        self.current_internal_loc = info.get('current_internal_loc', None)
        self.grid_info[7] = self.g_origin # reset grid state to top right corner
        # breakpoint()
        p = torch.relu(self.grid_info[2]@self.grid_info[7])
        return self.g_origin.squeeze(2), p, obs, info

    def convert_obs(self, obs, prev_action, same_as_prev_pos, force_prev=False):
        """
        Update representation given the action taken and/or sensory info using memory scaffold.
        """
        sinit = obs.unsqueeze(-1).unsqueeze(0)
        sinit = sinit.cuda() if self.use_cuda else sinit.numpy()

        obs, p, spatial_axis, evidence_direction = self.grid_step(sinit, prev_action, same_as_prev_pos, force_prev)
            
        return obs, p, spatial_axis, evidence_direction
        
    def set_env(self, env, k=0):
        # move grid cell to the initial position.
        env.current = self.env.current_position
        self.env = env
        # if False: # shift
        #     L = env.get_seq_len()
        #     Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps = self.grid_info

        #     direction = 1
        #     step = L + 1
        #     Wgg = Wggs[direction]
        #     for _ in range(step):
        #         g = Wgg @ g
           
        #     self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps]

        self.k = k

    def get_env(self):
        return self.env, self.k

    def store_env(self):
        """
            Store the current environmental information for simulation and recall.
        """
        self.env.store_env()
        self.fixed_pos = self.current_pos
        self.fixed_grid_info = copy.deepcopy(self.grid_info)

    def recall_env(self):
        self.env.recall_env()
        self.current_pos = self.fixed_pos
        self.grid_info = self.fixed_grid_info
        del self.fixed_pos
        del self.fixed_grid_info

    def step(self, action):
        """
        Proceed in current environment according to `action`.
        """
        obs, reward, done, info, _, _ = self.env.step(action)
        prev_pos = self.current_pos
        prev_internal_loc = self.current_internal_loc
        
        self.current_pos = info['current_pos']
        same_as_prev_pos = prev_pos == self.current_pos
        self.current_internal_loc = info.get('current_internal_loc', None)
        self.k += 1

        prev_action = info['prev_action']
        g, p, spatial_axis, evidence_axis = self.convert_obs(obs=obs, prev_action=prev_action, same_as_prev_pos=same_as_prev_pos)
        return g, p, obs, reward, done, info, spatial_axis, evidence_axis


    def grid_step(self, sinit, action, same_as_prev_pos, force_prev=False, recon_only=False):
        """
        sinit is the sensory input, 
        convert spatial info and evidence to internal indexing of the grid cell:
            - during process: 
                - if disp (action) == L or R, then 0 or 1
                - if disp (action) == straight, then 1
            - no need to care decision period since it's just one step
        """
        # Extract the latest left and right location info
        obs = sinit.squeeze()
        length = len(obs) // 2
        left_fov = obs[:length]
        right_fov = obs[length:]
        latest_obs = (int(left_fov[0]), int(right_fov[0]))
        
        # evidence_axis: convert latest obs to #right - #left to move in evidence axis 
        # if self.task_type == 'evidence-based':
        #     evidence_map = {(0,1):1, (1,0):-1, (1,1):0, (0,0):0, (-1,-1):0}
        # elif self.task_type in ['memory-based', 'context-based']:
        #     evidence_map = {(2,2):-1, (-2,-2):1, (0,0):0, (-1,-1):0, (1,0):0, (0,1):0}
        #     if self.task_type == 'context-based':
        #         context_map = {(1,0):-1, (0,1):1}
        #         # update context if NOT here cuz stuck and at second-to-last position
        #         context_direction = context_map[latest_obs] if ((not same_as_prev_pos) and (self.env.current_position == self.env.sequence_length - 2)) else 0 
        #         if latest_obs in [(0,1), (1,0)] and not (self.env.current_position == self.env.sequence_length - 2):
        #             breakpoint()
        #             assert 'tower should only appear second-to-last in a context-based task!'
                
        """update evidence axis ONLY if we have moved (same_as_prev_pos=False) and not at start"""
        # evidence_direction = 0 if (same_as_prev_pos and not self.k == 1) else evidence_map[latest_obs] '
        # breakpoint()
        evidence_direction = 0 if (same_as_prev_pos and not self.k == 1) else self.env.true_local_evidence[self.env.current_position]
            
        if self.dimension == 2:
            if self.grid_assignment is None or all(x == 'both' for x in self.grid_assignment):
                # spatial_axis: moving horizontally (L/R) is 0, moving vertically (forward) is 1
                if action in [0, 1]:
                    spatial_axis = 0 
                elif action == 2: # go straight
                    spatial_axis = 1
                """
                the only time evidence is updated is when you move (and new obs comes in)
                so evidence module should always be moving in axis=1
                """
                disp = (spatial_axis, evidence_direction) 
            else:
                disp = []
                for assignment in self.grid_assignment:
                    if assignment == 'position':
                        if action == 2 and self.env.current_position < self.env.sequence_length - 1:
                            # if in central arm and choose move forward
                            disp.append(1)
                        else:
                            disp.append(0)
                    elif assignment in ['both', 'evidence']:
                        disp.append(evidence_direction)
                    elif assignment == 'context':
                        assert self.task_type == 'context-based'
                        disp.append(context_direction)
                    else:
                        assert 'invalid entry in grid_assignment'
                disp = tuple(disp)
        else:
             assert 'TODO: disp configuration not implemented for 1d'       

        # NOTE: mental navigation 2d implementaiton is from water maze https://github.com/FieteLab/mental_navigation/tree/watermaze, gridUtils, gridWrapper
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps = self.grid_info
        gap = 3 + (getattr(self.env, 'mat_size', 0) // 2)
        if self.debug:
            print(reshape_to_square_tensors(split_tensor_by_lambdas(g, self.lambdas, n=self.dimension), self.lambdas, self.dimension))
        if recon_only:
            p, g, s = run_grid_cells_2d(sinit, disp, Wggs, Wgp, Wpg, 
                                Wsp, Wps, module_gbooks, module_sizes, g,
                                sbook_obs, pbook_obs, theta_sp, theta_ps, self.use_cuda, self.is_eval, force_prev, verbose=self.verbose, allocated_grids=self.allocated_grids, gap=gap, recon_only=True)
        else:
            p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells_2d(sinit, disp, Wggs, Wgp, Wpg, 
                                                                                     Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                     sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                     self.use_cuda, self.is_eval, force_prev, 
                                                                                     verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                     gap=gap)
            if all(x == 'both' for x in self.grid_assignment) and not same_as_prev_pos:
                """
                If NOT same_as_prev_pos, we have moved in space,
                this means we update evidence earlier;
                to complete the operation of `both`, 
                we update the position axis for all module here.
                Position module moves in axis=0, forward (1), thus action/disp here is (0,1).
                """
                p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells_2d(sinit, (0,1), Wggs, Wgp, Wpg, 
                                                                                     Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                     sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                     self.use_cuda, self.is_eval, force_prev, 
                                                                                     verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                     gap=gap)
            if not all(x == 'both' for x in self.grid_assignment) and 'both' in self.grid_assignment and not same_as_prev_pos:
                p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells_2d(sinit, 'trigger', Wggs, Wgp, Wpg, 
                                                                                     Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                     sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                     self.use_cuda, self.is_eval, force_prev, 
                                                                                     verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                     gap=gap)
            self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps]
        
        if type(g) is not torch.Tensor:
            g = torch.Tensor(g).float()
            p = torch.Tensor(p).float()

        if self.sigma > 0:
            nonzeros = g.nonzero()[:,1:2] # 1, N
            indices = torch.arange(g.shape[1], device=g.device).unsqueeze(0)
            d = ((indices - nonzeros) / self.sigma) ** 2
            prob = ((2 * torch.pi) ** -0.5) / self.sigma * torch.exp(- d / 2)
            g = prob.sum(0, keepdim=True).unsqueeze(-1)
        # breakpoint()
        # new_obs = []
        # new_obs.append(g[:,:,0])
        # g = torch.cat(new_obs, dim=1).float()
        # breakpoint()
        # if type(sinit) is np.ndarray:
        #     sinit = torch.from_numpy(sinit)

        # if force_prev:
        #     grids, places, sensories = self.gps
        #     correct_grid = grids[(sensories == sinit).all(1)[:,0]][0:1]
        #     correct_place = places[(sensories == sinit).all(1)[:,0]][0:1]
        #     check_g = ((g == correct_grid).all())
        #     check_p = ((p == correct_place).all())
        #     if not check_g:
        #         print(check_g, check_p)
        #         breakpoint()
        #         raise Exception("Reconstruction is failed")
        if self.debug:
            print(reshape_to_square_tensors(split_tensor_by_lambdas(g, self.lambdas, n=self.dimension), self.lambdas, self.dimension))
            breakpoint()
        return g.squeeze(2), p, action, evidence_direction