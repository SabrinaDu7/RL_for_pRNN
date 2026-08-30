import torch
import numpy as np

from curious_george.rl.collect.format import get_obss_preprocessor


class ActorCriticAgent:
    """Samples actions from the AC model using pRNN-derived SRs.

    Device policy: the agent follows the AC model's parameters (`self.device`
    is dynamic). `getObservations` aligns the pRNN to that device on entry and
    leaves it there - it never moves models behind the caller's back. Callers
    that need a specific device (e.g. CPU for numpy-based analysis) wrap the
    call in `curious_george.models.device.on_device`.
    """

    def __init__(self, action_space, acmodel, prnn, device, argmax: bool, pastSR=True):
        self.action_space = action_space
        self.acmodel = acmodel
        self.prnn = prnn
        self.pastSR = pastSR
        self.argmax = argmax
        self.name = "ActorCritic Agent"

    @property
    def device(self) -> torch.device:
        """The AC model's current device (moves with on_device contexts)."""
        return next(self.acmodel.parameters()).device

    def next_SR(self, obs, act):
        obs = [obs, obs]

        obs_pN, act_pN = self.prnn.env_shell.env2pred(obs, act)
        obs_pN, act_pN = obs_pN.to(self.device), act_pN.to(self.device)
        with (
            torch.no_grad()
        ):  # calculate SR for step t based on obs and action from step t-1
            SR = self.prnn.predict_single(obs_pN[:, :-1, :], act_pN).squeeze(dim=0)

        return SR

    def getObservations(self,
                        env,
                        tsteps,
                        reset=True,
                        includeRender=False,
                        start_pos: tuple | None = None,
                        start_dir: int | None = None,
                        **kwargs):
        self.prnn.pRNN.to(self.device)
        render = False

        obs = [None for t in range(tsteps + 1)]
        act = [None for t in range(tsteps)]

        if reset:
            obs[0] = env.reset()

        else:
            o = env.env.gen_obs()
            obs[0] = env.env.observation(o)

        if start_pos is not None or start_dir is not None:
            # These were DECLARED and never read: every caller that passed them
            # silently got a random start. Placing the agent is the same
            # assignment `envs/obs_bank.py` uses to enumerate the room, and the
            # observation has to be regenerated afterwards or obs[0] still shows
            # the view from wherever reset() happened to land.
            u = env.env.unwrapped
            if start_pos is not None:
                x, y = (int(v) for v in start_pos)
                cell = u.grid.get(x, y)
                if not (0 <= x < u.width and 0 <= y < u.height) or (
                    cell is not None and not cell.can_overlap()
                ):
                    raise ValueError(
                        f"start_pos {(x, y)} is not a standable cell of this "
                        f"{u.width}x{u.height} room"
                    )
                u.agent_pos = (x, y)
            if start_dir is not None:
                if not 0 <= int(start_dir) < 4:
                    raise ValueError(f"start_dir must be 0..3, got {start_dir}")
                u.agent_dir = int(start_dir)
            obs[0] = env.env.observation(env.env.gen_obs())

        state = {
            "agent_pos": np.resize(env.get_agent_pos(), (1, 2)),
            "agent_dir": env.get_agent_dir(),
        }

        if includeRender:
            render = [None for t in range(tsteps + 1)]
            render[0] = env.render(mode=None)

        # TODO: Double check with Alex
        SR = torch.zeros((1, self.prnn.hidden_size), device=self.device)
        state["SRs"] = SR.cpu().numpy()

        _, preprocess_obss = get_obss_preprocessor(env.observation_space)
        for t in range(tsteps):
            preprocessed_obs = preprocess_obss([obs[t]], device=self.device)
            with torch.no_grad():
                dist, _ = self.acmodel(preprocessed_obs, SR=SR)

                if self.argmax:
                    actions = dist.probs.max(1)[1]
                else:
                    actions = dist.sample()

                act[t] = actions.cpu().numpy()

            obs[t + 1] = env.step(act[t])[0]
            state["agent_pos"] = np.append(
                state["agent_pos"], np.resize(env.get_agent_pos(), (1, 2)), axis=0
            )
            state["agent_dir"] = np.append(state["agent_dir"], env.get_agent_dir())

            if self.pastSR:
                SR = self.next_SR(obs=obs[t], act=act[t]) # obs that lead to the action
            else:
                SR = self.next_SR(obs=obs[t + 1], act=act[t])

            state["SRs"] = np.append(state["SRs"], SR.cpu().numpy())

            if includeRender:
                render[t + 1] = env.render(mode=None)

        act = np.array(act).reshape(-1)
        return obs, act, state, render
