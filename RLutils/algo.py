# Based on the PPO algo from torch-ac library (https://github.com/lcswillems/torch-ac)


import torch
import numpy

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from scipy.spatial.distance import cosine


class PredictivePPOAlgo:
    """The base class for RL algorithms."""

    def __init__(self, env, acmodel, predictiveNet=None, device=None, num_frames=None, discount=0.99, lr=0.001,
                 gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None, place_cells=None,
                 cann=None, train_pN=False, noise_mu=0, noise_std=0.03, intrinsic=False, k_int=1,
                 reshape_reward=None):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        env : environment
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time - NO?
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle - NO?
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        print('Store parameters')
        self.env = env
        self.acmodel = acmodel
        self.pN = predictiveNet
        self.device = device
        self.num_frames = num_frames or 128
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.intrinsic = intrinsic
        self.k_int = k_int
        self.PC = place_cells
        self.CANN = cann
        self.train_pN = train_pN
        self.noise_mu = noise_mu
        self.noise_std = noise_std

        # Control parameters
        print('Control parameters')
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames % self.recurrence == 0

        # Configure models
        print('Configure acmodel')
        self.acmodel.to(self.device)
        self.acmodel.train()
        if self.pN:
        # TODO: should be elsewhere if saving the net
            self.pN.pRNN.to(self.device)

        self.obs = self.env.reset()
        self.loc = self.env.get_agent_pos()
        print('Reset done')
        self.obss = [None] * (self.num_frames)
        self.locs = [None] * (self.num_frames)
        if self.acmodel.recurrent:
            self.memory = torch.zeros((1,self.acmodel.memorysize), device=self.device)
            self.memories = torch.zeros(self.num_frames, self.acmodel.memorysize, device=self.device)
        self.mask = 1
        self.masks = torch.zeros(self.num_frames, device=self.device)
        print('Masks done')
        self.actions = torch.zeros(self.num_frames, device=self.device, dtype=torch.int)
        self.values = torch.zeros(self.num_frames, device=self.device)
        print('Values done')
        self.rewards = torch.zeros(self.num_frames, device=self.device)
        self.advantages = torch.zeros(self.num_frames, device=self.device)
        print('Advantages done')
        self.log_probs = torch.zeros(self.num_frames, device=self.device)
        self.int_rewards = torch.zeros(self.num_frames, device=self.device)

        # Initialize spatial representations (if used)
        self.SR = self.init_SR()
        self.SRs = torch.zeros((self.num_frames, self.SR.shape[1]), device=self.device)

        # Initialize log values
        print('Initialize log values')
        self.log_episode_return = 0
        self.log_episode_reshaped_return = 0
        self.log_episode_num_frames = 0

        self.log_done_counter = 0
        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        # Initialize intrinsic rewards
        if self.intrinsic:
            self.update_ref(None)
            self.int_rewards = torch.zeros(self.num_frames, device=self.device)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        print('All done')

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames, ...).
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames):
            # Do one agent-environment interaction

            SR, action, dist, value, memory, det_action = self.next_experience()

            obs, reward, terminated, truncated, _ = self.env.step(det_action)
            loc = self.env.get_agent_pos()
            done = terminated or truncated

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.locs[i] = self.loc
            self.loc = loc
            self.SRs[i] = self.SR # SR at step i is from step i-1 (empty at step 0)
            self.SR = SR
            # if self.acmodel.recurrent: # Not using it now, disabled for efficiency
            #     self.memories[i] = self.memory
            #     self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - done
            self.actions[i] = action
            self.values[i] = value
            # if self.reshape_reward is not None: # Not using it now, disabled for efficiency
            #     self.rewards[i] = self.reshape_reward(obs, action, reward, done)
            # else:
            #     self.rewards[i] = reward
            self.rewards[i] = reward
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += reward
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += 1

            if done:
                if self.intrinsic and reward > 1e-5:
                    SR,_,_,_,_,_ = self.next_experience()
                    self.update_ref(SR)
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return)
                self.log_reshaped_return.append(self.log_episode_reshaped_return)
                self.log_num_frames.append(self.log_episode_num_frames)
                if self.pN:
                    self.pN.reset_state(device=self.device)
                self.SR = self.init_SR()
                self.obs = self.env.reset()
                self.log_episode_return = 0
                self.log_episode_reshaped_return = 0
                self.log_episode_num_frames = 0

        # Calculate intrinsic rewards
        if self.intrinsic:
            SR,_,_,_,_,_ = self.next_experience()
            # Add SR from the last state
            SRs = torch.cat((self.SRs, SR), dim=0).cpu()
            # Calculate cosine similarity between SRs and reference SR
            errors = torch.tensor([cosine(SR, self.ref.squeeze().cpu()) for SR in SRs[1:]], device=self.device)
            errors = torch.cat((errors[0][None], errors), dim=0)
            # Calculate intrinsic rewards
            self.int_rewards = errors[:-1] - errors[1:]

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
                # TODO: remove SR if not using PCs with recurrence
                _, next_value, _ = self.acmodel(preprocessed_obs,
                                                (self.memory * self.mask)[None],
                                                noise=noise,
                                                SR=self.SR)
            else:
                _, next_value = self.acmodel(preprocessed_obs, SR=self.SR)

        for i in reversed(range(self.num_frames)):
            next_mask = self.masks[i+1] if i < self.num_frames - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames - 1 else 0

            delta = self.rewards[i] + self.k_int * self.int_rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        exps = DictList()
        exps.obs = self.obss
        exps.SR = self.SRs
        if self.acmodel.recurrent:
            exps.memory = self.memories
            exps.mask = self.masks
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage # approximates current and discounted future returns
        exps.log_prob = self.log_probs

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Reset pN state
        if self.pN:
            self.pN.reset_state(device=self.device)

        # Log some values

        logs = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": self.log_reshaped_return,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames,
            "num_episodes": self.log_done_counter,
            "intrinsic_rewards": self.int_rewards.tolist()
        }

        self.log_return = []
        self.log_reshaped_return = []
        self.log_num_frames = []

        return exps, logs


    def update_parameters(self, exps):
        # Collect experiences
        # TODO: deal with analyses, predNet saving, option to backprop through pRNN

        for _ in range(self.epochs): # TODO: should it be just one epoch?
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
                        # NOTE: it is still not analogous to pRNN because obs is from current step, and act is not provided
                        dist, value, memory = self.acmodel(sb.obs, (memory.T * sb.mask).T[None], noise=noise, SR=sb.SR)
                    else:
                        dist, value = self.acmodel(sb.obs, SR=sb.SR)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Update pN

        if self.train_pN:
            obs, act = self.pN.env_shell.env2pred(exps.obs + [self.obs], exps.action) # including the last observation
            _,_,_ = self.pN.trainStep(obs, act, mask=exps.masks+[self.mask])

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs


    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
    

    def next_experience(self):
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        memory = None

        with torch.no_grad():
            if self.acmodel.recurrent:
                noise = self.noise_mu + self.noise_std * torch.randn(self.acmodel.memorysize, device=self.device)
                # TODO: remove SR if not using PCs with recurrence
                dist, value, memory = self.acmodel(preprocessed_obs,
                                                    (self.memory * self.mask)[None],
                                                    noise=noise,
                                                    SR=self.SR)
            else:
                dist, value = self.acmodel(preprocessed_obs, SR=self.SR)
        action = dist.sample() # choose action based on SR from step t-1
        det_action = action.cpu().numpy()

        if self.pN:
            obs_pN, act_pN = self.pN.env_shell.env2pred([self.obs,self.obs], det_action)
            obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
            with torch.no_grad(): # calculate SR for step t based on obs and action from step t-1
                SR = self.pN.predict_single(obs_pN[:,:-1,:], act_pN).squeeze(dim=0)
        elif self.PC: # not calculating it on the same step for now
            SR = torch.tensor(self.PC.activation(self.env.agent_pos), dtype=torch.float32, device=self.device).unsqueeze(dim=0)
        elif self.CANN: # not calculating it on the same step for now
            with torch.no_grad():
                _,_,SR = self.CANN.predict(preprocessed_obs,None,self.env.get_agent_pos())
            SR = SR[0].to(self.device)
        else:
            SR = torch.tensor([], device=self.device).unsqueeze(dim=0)

        return SR, action, dist, value, memory, det_action
    

    def init_SR(self):
        SR = torch.tensor([], device=self.device).unsqueeze(dim=0)
        if self.pN:
            # TODO: change for theta
            SR = torch.zeros((1,self.pN.hidden_size), device=self.device)
        if self.CANN:
            SR = torch.zeros((1,self.CANN.hidden_size), device=self.device)
        elif self.PC:
            SR = torch.zeros((1,self.PC.size), device=self.device)
        return SR
    

    def update_ref(self, activations):
        if hasattr(self, 'ref'):
            self.nrefs += 1
            self.ref = self.ref + (activations - self.ref) / self.nrefs
        else:
            self.ref = torch.zeros((1,self.SR.shape[1]), device=self.device)
            self.nrefs = 1