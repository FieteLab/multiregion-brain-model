import numpy as np
import torch
import copy
import random

from VectorHaSH.gridUtils import create_binary_mask, generate_bernoulli_tensor, initialize_p_masks, grid_cell_initial_2d, run_grid_cells_2d, run_grid_cells_2d_new_model
from VectorHaSH.base_wrapper import BaseWrapper
from VectorHaSH.grid_cells.assoc_utils_np import train_gcpc, pseudotrain_3d_iterative, pseudotrain_3d_iterative_step, module_wise_NN, pseudotrain_Wps, pseudotrain_Wsp, gen_gbook
from towertask.utils import split_tensor_by_lambdas, reshape_to_square_tensors, set_seed

class GridNew2dRecurWrapper(BaseWrapper):
    def __init__(self, env, arg_gcpc, mode='gp', sigma=0, use_cuda=True, 
                 grid_info=None, convert_target=False, dimension=2, conv_int=True, 
                 grid_step_size=1, grid_assignment=['position', 'position', 'evidence'],
                 debug=False, is_eval=False, task_type='evidence-based', new_model=True, with_mlp=False, args=None):
        super().__init__(env)
        self.task_type = task_type # evidence-based (tower), context-based, memory-based
        self.use_cuda = use_cuda
        self.current_internal_loc = None
        self.dimension = dimension
        self.grid_step_size = grid_step_size
        self.grid_assignment = grid_assignment
        self.is_eval = is_eval
        self.new_model = new_model
        self.with_mlp = with_mlp
        assert args != None
        self.args = args
        
        assert self.args.add_recurrence, 'recurrence not included'

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
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps, Wpp, theta_pp = grid_cell_initial_2d(self.Np, self.Ns, self.lambdas, 
                                                                                                                                    self.use_cuda, self.dimension, 
                                                                                                                                    self.grid_assignment,
                                                                                                                                    self.new_model, self.with_mlp, 
                                                                                                                                    with_recurrence=True)
        # initialization for new model
        # self.mask_pg, self.mask_ps = torch.tensor(create_binary_mask(Wps, percentage=50), dtype=torch.float32, device=Wgp.device) # only 15% of the population is active
        self.mask_pg, self.mask_ps = initialize_p_masks(self.Np, 0.5, 0.5, device=Wgp.device) 
        print("mask_pg and mask_ps are deprecated and never used--this needs to be removed.")
        self.W_L = torch.eye(self.Ns//2, dtype=torch.float32, device=Wgp.device).unsqueeze(0)
        self.W_R = torch.eye(self.Ns//2, dtype=torch.float32, device=Wgp.device).unsqueeze(0)

        self.g_origin = g
        torch.cuda.empty_cache()
        self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps, self.mask_pg, self.mask_ps, self.W_L, self.W_R, 0, Wpp, theta_pp]
        if grid_info is not None:
            print('***Loaded grid info!***')
            self.grid_info = grid_info
            self.grid_info[7] = self.g_origin

    def reset(self, episode_idx=0):
        obs, info = self.env.reset(episode_idx)
        # breakpoint()
        """TODO: Need to check with Jaedong on what to use for `prev_action` and `same_as_prev_pos`"""
        # since this reset() gives obs at position 0, we should update evidence axis by setting `same_as_prev_pos` False.
        # obs, p, _, _ = self.convert_obs(obs, prev_action=2, same_as_prev_pos=False, force_prev=self.k > 0)
        # update the current position
        self.current_pos = info['current_pos']
        self.current_internal_loc = info.get('current_internal_loc', None)
        self.grid_info[7] = self.g_origin # reset grid state to top right corner
        self.same_as_prev_pos = False
        Wpg, W_L, W_R = self.grid_info[2], self.grid_info[14], self.grid_info[15]
        Wsp, Wps, theta_sp, theta_ps = self.grid_info[3], self.grid_info[4], self.grid_info[10], self.grid_info[11]
        Wpp, theta_pp = self.grid_info[17], self.grid_info[18]
        
        length = len(obs) // 2
        W = torch.zeros((2 * length, 2 * length), device=W_L.device)
        W[:length, :length] = W_R
        W[length:, length:] = W_L
        sensory = W @ obs.view(1, 2 * length, 1).cuda()
        p_s = Wps @ sensory
        self.grid_info[16] = p_s
        p_g = Wpg @ self.g_origin

        if self.new_model:
            p = torch.relu(p_g+p_s)
        else:
            p = torch.relu(p_g)
        self.prev_p = torch.zeros_like(p, device='cuda')
        # Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0].cuda(), use_torch=True)
        # Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0].cuda(), p[:,:,0], use_torch=True)
        # self.grid_info[3], self.grid_info[4], self.grid_info[10], self.grid_info[11] =  Wsp, Wps, theta_sp, theta_ps
        # p = torch.relu(self.grid_info[2]@self.grid_info[7])
        return self.g_origin.squeeze(2), p, obs, info, p_s#torch.relu(self.mask_ps*p_s)

    def convert_obs(self, obs, prev_action, same_as_prev_pos, force_prev=False):
        """
        Update representation given the action taken and/or sensory info using memory scaffold.
        """
        sinit = obs.unsqueeze(-1).unsqueeze(0)
        sinit = sinit.cuda() if self.use_cuda else sinit.numpy()

        obs, p, spatial_axis, evidence_direction, p_s = self.grid_step(sinit, prev_action, same_as_prev_pos, force_prev)
            
        return obs, p, spatial_axis, evidence_direction, p_s
        
    def set_env(self, env, k=0):
        # move grid cell to the initial position.
        env.current = self.env.current_position
        self.env = env
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
        self.same_as_prev_pos = same_as_prev_pos
        self.current_internal_loc = info.get('current_internal_loc', None)
        self.k += 1

        prev_action = info['prev_action']
        g, p, spatial_axis, evidence_axis, p_s = self.convert_obs(obs=obs, prev_action=prev_action, same_as_prev_pos=same_as_prev_pos)
        return g, p, obs, reward, done, info, spatial_axis, evidence_axis, p_s


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
        
        """update evidence axis ONLY if we have moved (same_as_prev_pos=False) and not at start"""
        # evidence_direction = 0 if (same_as_prev_pos and not self.k == 1) else evidence_map[latest_obs] 
        # breakpoint()
        evidence_direction = 0 if (same_as_prev_pos and not self.k == 1) or (self.env.current_position > self.env.sequence_length - 1) else self.env.true_local_evidence[self.env.current_position]
            
        if self.dimension == 2:
            # get spatial axis; 1 if move forward in corridor, 0 otherwise (i.e. stuck in corridor or reached the end)
            if action == 2 and self.env.current_position < self.env.sequence_length - 1:
                spatial_axis = 1
            else:
                spatial_axis = 0
                
            if self.grid_assignment is None or all(x == 'both' for x in self.grid_assignment):
                # spatial_axis: moving horizontally (L/R) is 0, moving vertically (forward) is 1
                # if action in [0, 1]:
                #     spatial_axis = 0 
                # elif action == 2: # go straight
                #     spatial_axis = 1
                """
                the only time evidence is updated is when you move (and new obs comes in)
                so evidence module should always be moving in axis=1
                """
                disp = (spatial_axis, evidence_direction) 
            else:
                disp = []
                for assignment in self.grid_assignment:
                    if assignment == 'position':
                        disp.append(spatial_axis)
                        # if action == 2 and self.env.current_position < self.env.sequence_length - 1:
                            # if in central arm and choose move forward
                            # spatial_axis = 1
                            # disp.append(1)
                        # else:
                        #     spatial_axis = 0
                        #     disp.append(0)
                    elif assignment in ['both', 'evidence']:
                        disp.append(evidence_direction)
                    else:
                        assert 'invalid entry in grid_assignment'
                disp = tuple(disp)
        else:
             assert 'TODO: disp configuration not implemented for 1d'       

        # NOTE: mental navigation 2d implementaiton is from water maze https://github.com/FieteLab/mental_navigation/tree/watermaze, gridUtils, gridWrapper
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps, mask_pg, mask_ps, W_L, W_R, p_s, Wpp, theta_pp = self.grid_info
        gap = 3 + (getattr(self.env, 'mat_size', 0) // 2)
        if self.debug:
            print('before grid_step', reshape_to_square_tensors(split_tensor_by_lambdas(g, self.lambdas, n=self.dimension), self.lambdas, self.dimension))
        if recon_only:
            p, g, s = run_grid_cells_2d(sinit, disp, Wggs, Wgp, Wpg, 
                                Wsp, Wps, module_gbooks, module_sizes, g,
                                sbook_obs, pbook_obs, theta_sp, theta_ps, self.use_cuda, 
                                self.is_eval, force_prev, verbose=self.verbose, 
                                allocated_grids=self.allocated_grids, gap=gap, recon_only=True)
        else:
            """
            Note that I set is_eval to true here to avoid Wps, Wsp, Wpp updates.
            """
            if not self.is_eval: # at training
                this_eval = self.with_mlp # if use mlp, then do eval, else this is False (no mlp) so we update the weights
            else:
                this_eval = self.is_eval
            # breakpoint()
            p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps, p_s, Wpp, theta_pp = run_grid_cells_2d_new_model(sinit, disp, Wggs, Wgp, Wpg, 
                                                                                                Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                                sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                                self.use_cuda, this_eval, force_prev, #is_eval set to True to avoid updates
                                                                                                verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                                gap=gap, W_L=W_L, W_R=W_R, mask_pg=mask_pg, mask_ps=mask_ps, 
                                                                                                new_model=self.new_model, with_mlp=self.with_mlp, 
                                                                                                with_recurrence=True, Wpp=Wpp, theta_pp=theta_pp, prev_p=self.prev_p)
            if all(x == 'both' for x in self.grid_assignment) and not same_as_prev_pos:
                """
                If NOT same_as_prev_pos, we have moved in space,
                this means we update evidence earlier;
                to complete the operation of `both`, 
                we update the position axis for all module here.
                Position module moves in axis=0, forward (1), thus action/disp here is (0,1).
                """
                assert False, 'check behavior in new model before calling'
                p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells_2d(sinit, (0,1), Wggs, Wgp, Wpg, 
                                                                                     Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                     sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                     self.use_cuda, self.is_eval, force_prev, 
                                                                                     verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                     gap=gap)
            if not all(x == 'both' for x in self.grid_assignment) and 'both' in self.grid_assignment and not same_as_prev_pos:
                assert False, 'check behavior in new model before calling'
                p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells_2d(sinit, 'trigger', Wggs, Wgp, Wpg, 
                                                                                     Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                     sbook_obs, pbook_obs, theta_sp, theta_ps, 
                                                                                     self.use_cuda, self.is_eval, force_prev, 
                                                                                     verbose=self.verbose, allocated_grids=self.allocated_grids, 
                                                                                     gap=gap)
            self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps, mask_pg, mask_ps, W_L, W_R, p_s, Wpp, theta_pp]
        
        if type(g) is not torch.Tensor:
            g = torch.Tensor(g).float()
            p = torch.Tensor(p).float()

        if self.sigma > 0:
            nonzeros = g.nonzero()[:,1:2] # 1, N
            indices = torch.arange(g.shape[1], device=g.device).unsqueeze(0)
            d = ((indices - nonzeros) / self.sigma) ** 2
            prob = ((2 * torch.pi) ** -0.5) / self.sigma * torch.exp(- d / 2)
            g = prob.sum(0, keepdim=True).unsqueeze(-1)

        new_obs = []
        # let obs just be g
        # if 'g' in self.mode:
        new_obs.append(g[:,:,0])
        # if 'p' in self.mode:
        #     new_obs.append(p[:,:,0])
        # if 's' in self.mode:
        #     new_obs.append(s[:,:,0])

        obs = torch.cat(new_obs, dim=1).float()
        if type(sinit) is np.ndarray:
            sinit = torch.from_numpy(sinit)

        if self.sanity_mode:
            return obs, p, g, s, sinit
        
        if self.debug:
            print('after grid_step', reshape_to_square_tensors(split_tensor_by_lambdas(g, self.lambdas, n=self.dimension), self.lambdas, self.dimension))
            breakpoint()
        if torch.isnan(p).any():
            p = torch.tensor([x if x is not None else 0 for x in p], dtype=torch.float32)
            
        self.prev_p = p.clone()
        return obs, p, spatial_axis, evidence_direction, p_s # the first one is just g