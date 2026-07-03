import torch
import numpy as np

from curious_george.rl.format import get_obss_preprocessor



class ActorCriticAgent:
    def __init__(self, action_space, acmodel, prnn, device, argmax: bool, pastSR=True):
        self.action_space = action_space
        self.acmodel = acmodel
        self.prnn = prnn
        self.device = device
        self.pastSR = pastSR
        self.argmax = argmax
        self.name = "ActorCritic Agent"
        assert pastSR is not ("prevAct" in str(prnn.pRNN))

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

        for t in range(tsteps):
            # obs_tensor = torch.tensor(obs[aa]['image'], device=self.device)
            _, preprocess_obss = get_obss_preprocessor(env.observation_space)
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

        self.prnn.pRNN.to("cpu") # LANDMINE: WHERE SETTING TO CPU AGAIN???

        act = np.array(act).reshape(-1)
        return obs, act, state, render
