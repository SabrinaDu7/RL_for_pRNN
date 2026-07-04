import warnings

### Suppress warnings ###
warnings.filterwarnings("ignore", category=UserWarning)

import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from omegaconf import OmegaConf, DictConfig
import hydra
import wandb
from tqdm import tqdm

from utils import get_ckpt_env_vars, get_wandb_env_vars, StatusCkptKeys, AgentType
import curious_george as RLutils
from curious_george import (
    DEVICE,
    ACModel,
    ACModelSR,
    ActorCriticAgent,
    PredictivePPOAlgo,
    OnPolicyAnalysis,
    mutual_info_policy,
    get_agent,
)
from curious_george.world_model.device import on_device
from curious_george.evaluation.spatial import evaluate_spatial_representation
from curious_george.rl.collector import BatchedPredictivePPOAlgo

from prnn.utils import (
    PredictiveNet,
    RandomActionAgent,
    load_pN,
    save_pN,
)
from utils import load_statedict_from_acmodel_status

class RL_Trainer(object):
    def __init__(self, params):

        self.params = params
        self.group = params.exp.exp_name
        self.wandb_log = self.params.logging.wandb_log

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        agent_type = "curious" if params.exp.curious_agent else "rand"
        run_name = f"{self.group}_{agent_type}_{date}"
        self.model_name = f"{run_name}/"

        # Set run dir
        self.model_dir = RLutils.get_model_dir(self.model_name)
        RLutils.create_folders_if_necessary(self.model_dir)

        if params.logging.video_log_freq != 0:
            self.video_dir = RLutils.get_video_dir(self.model_name)
            RLutils.create_folders_if_necessary(self.video_dir)
        else:
            self.video_dir = ""

        print("\n\n\nLOGGING TO: ", self.model_dir, "\n\n\n")

        if self.wandb_log:
            self.run = wandb.init(
                entity=params.logging.wandb_entity,
                project=params.logging.wandb_project,
                group=self.group,
                name=run_name,
                id=run_name,
                dir=self.model_dir,
                resume="allow",
                config=OmegaConf.to_container(params, resolve=True), # type: ignore
            )

    def run_training_loop(self):
        args = self.params

        RLutils.seed(args.exp.seed)

        print(f"Device: {DEVICE}\n")

        # Load environment (default size = 16)
        agent_start_pos = None
        agent_start_dir = None
        start_room = None

        if not args.exp.start_rand:
            start_room = args.exp.start_room
            print(f"Agent starting in room {start_room}")

        env = RLutils.make_env(
            env_key=args.exp.env_name,
            input_type=args.exp.input_type,
            seed=args.exp.seed + 10000,
            act_enc=args.predNet.action_encoding,
            open_all_paths=False, # Only applicable for FourRooms env
            subroom_size=args.exp.env_subroom_size, # Only applicable for FourRooms env
            door_poss=args.exp.door_poss, # Only applicable for FourRooms env
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            agent_start_room=start_room, # Only applicable for FourRooms env
        )
        print("Environment loaded\n")

        # Load training status

        if args.logging.load_acmodel:
            prnn_ckpt, acmodel_status_ckpt = get_ckpt_env_vars()
            status = torch.load(
                acmodel_status_ckpt,
                map_location=DEVICE,
                weights_only=False
            )
        else:
            status = {StatusCkptKeys.NUM_FRAMES.value: 0, StatusCkptKeys.UPDATE.value: 0}
        print("Training status loaded\n")

        # Load observations preprocessor

        obs_space, preprocess_obss = RLutils.get_obss_preprocessor(
            env.observation_space
        )
        # if "vocab" in status:
        #     preprocess_obss.vocab.load_vocab(status["vocab"])
        print("Observations preprocessor loaded\n")

        # Load pRNN
        predictiveNet = PredictiveNet(
            env,
            hidden_size=args.predNet.hiddensize,
            pRNNtype=args.predNet.pRNNtype,
            learningRate=args.predNet.lr,
            bptttrunc=args.predNet.bptttrunc,
            weight_decay=args.predNet.weight_decay,
            neuralTimescale=args.predNet.ntimescale,
            dropp=args.predNet.dropout,
            trainNoiseMeanStd=(args.predNet.noisemean, args.predNet.noisestd),
            f=args.predNet.sparsity,
            wandb_log=self.wandb_log,
        )

        args.predNet.hiddensize = predictiveNet.hidden_size
        # predictiveNet.pRNN.to(device)
        predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0, 0])  # TODO: remove later
        # TODO: I think the above line is already set by default to [-1, 1, 0, 0] in FaramaMinigridShell(GymMinigrid)

        # Load pRNN
        if args.logging.load_worldmodel:
            prnn_ckpt, acmodel_status_ckpt = get_ckpt_env_vars()
            load_pN(model_ckpt_filepath=prnn_ckpt, 
                    device=DEVICE,
                    pRNNtype=args.predNet.pRNNtype, 
                    predictive_net=predictiveNet)
            
            print("\n" + "=" * 10)
            print(f"Existing pRNN model found at {prnn_ckpt} and loaded from state dict")
            print("=" * 10 + "\n")

        print("pRNN model initialized\n")
        prnn_eval_bool = args.exp.offpolicy_prnn_eval or args.exp.onpolicy_prnn_eval

        # Load ACModel
        acmodel: nn.Module
        if args.exp.pRNN:
            acmodel = ACModelSR(
                obs_space,
                env.action_space,
                args.predNet.hiddensize,
                args.exp.with_obs,
                args.exp.rgb,
                args.exp.with_HD,
            )

        else:
            acmodel = ACModel(
                obs_space, env.action_space, args.exp.with_HD, args.exp.rgb
            )

        if StatusCkptKeys.MODEL_STATE.value in status:
            print("\n" + "=" * 10)
            load_statedict_from_acmodel_status(
                receiver=acmodel,
                status=status,
                status_key=StatusCkptKeys.MODEL_STATE,
                device=DEVICE,
            )
            print(f"Existing AC model found.")
            print("=" * 10 + "\n")

        acmodel.to(DEVICE)
        print("AC model loaded\n")

        if args.predNet.train:
            assert args.predNet.seqdur > 0, "Set an appropriate seqdur"
        else:
            args.predNet.seqdur = 0

        # Load algo
        pastSR = not ("prevAct" in str(predictiveNet.pRNN))
        print("pastSR:", pastSR)

        num_envs = args.exp.get("num_envs", 1)
        if num_envs > 1:
            extra_envs = [
                RLutils.make_env(
                    env_key=args.exp.env_name,
                    input_type=args.exp.input_type,
                    seed=args.exp.seed + 10000 + 1000 * (i + 1),
                    act_enc=args.predNet.action_encoding,
                    open_all_paths=False,
                    subroom_size=args.exp.env_subroom_size,
                    door_poss=args.exp.door_poss,
                    agent_start_pos=agent_start_pos,
                    agent_start_dir=agent_start_dir,
                    agent_start_room=start_room,
                )
                for i in range(num_envs - 1)
            ]
            AlgoCls = BatchedPredictivePPOAlgo
            first_arg = [env] + extra_envs
        else:
            AlgoCls = PredictivePPOAlgo
            first_arg = env

        algo = AlgoCls(
            first_arg,
            acmodel,
            predictiveNet,
            DEVICE,
            args.rl.frames,
            args.rl.discount,
            args.rl.lr,
            args.rl.gae_lambda,
            args.rl.entropy_coef,
            args.rl.value_loss_coef,
            args.rl.max_grad_norm,
            1,  # recurrence (recurrent path removed)
            args.rl.optim_eps,
            args.rl.ppo_clip_eps,
            args.rl.ppo_epochs,
            args.rl.ppo_batch_size,
            preprocess_obss,
            args.predNet.train,
            args.predNet.noisemean,
            args.predNet.noisestd,
            args.predNet.seqdur,
            args.exp.intrinsic,
            args.rl.k_int,
            pastSR,
            args.exp.curious_agent,
            args.rl.k_curious,
            reward_alignment=args.rl.reward_alignment,
        )

        if StatusCkptKeys.OPTIMIZER_STATE.value in status:
            load_statedict_from_acmodel_status(
                receiver=algo.optimizer,
                status=status,
                status_key=StatusCkptKeys.OPTIMIZER_STATE,
                device=DEVICE,
            )
            print("Optimizer loaded\n")

        # Create random agent for analysis

        action_probability = np.array([0.15, 0.15, 0.6, 0.1])
        randomagent = get_agent(env=env, rand_act_prob=action_probability, agent_Type=AgentType.RANDOM)
        ac_agent = get_agent(env=env, agent_Type=AgentType.AC, prnn=predictiveNet, device=DEVICE, ac_model=acmodel, pastSR=pastSR)

        # Train model
        num_frames = status[StatusCkptKeys.NUM_FRAMES.value]
        update = status[StatusCkptKeys.UPDATE.value]
        start_time = time.time()
        header = False

        n_performance = 0
        error_map = None
        
        with tqdm(total=args.rl.steps, desc="Processing") as pbar:
            while num_frames < args.rl.steps: # num_frames' granularity is steps. It represents the number of steps taken in the env 
                # Update model parameters
                update_start_time = time.time()

                if args.exp.random_action_agent:
                    logs = algo.randomAgent_collect_exp_and_update(randomagent)
                else:
                    exps, logs1 = algo.collect_experiences()
                    logs2 = algo.update_parameters(
                        exps=exps, update_params=(not args.exp.random_init_control)
                    )
                    logs = {**logs1, **logs2}

                update_end_time = time.time()

                num_frames += logs["num_frames"]
                update += 1 # Update represents the number of 2048 steps taken. One update is the collection of 2048 steps (as seen in collect experiences)
                pbar.update(logs["num_frames"])  # was never updated before (bar stuck at 0%)

                # Print logs

                if update % args.logging.log_interval == 0:
                    fps = logs["num_frames"] / (update_end_time - update_start_time)
                    duration = int(time.time() - start_time)
                    num_frames_per_episode = RLutils.synthesize(
                        logs["num_frames_per_episode"]
                    )

                    # plotSampleTrajectory runs predict on CPU tensors, so pin
                    # the models to CPU for the call; placement is restored on exit.
                    with on_device([predictiveNet, acmodel], "cpu"):
                        predictiveNet.plotSampleTrajectory(
                                env=env,
                                agent=ac_agent,
                            ) # Logs to wandb inside the function if predictiveNet.wandb_log is True

                    if not args.exp.random_action_agent:
                        return_per_episode = RLutils.synthesize(
                            logs["return_per_episode"], signs=True
                        )
                        int_rewards = RLutils.synthesize(
                            logs["intrinsic_rewards"], abs=True
                        )
                        cur_rewards = RLutils.synthesize(logs["curious_rewards"], abs=True)
                        values = RLutils.synthesize(logs["values"])
                        advantages = RLutils.synthesize(logs["advantages"])

                    if not header:
                        header = ["update"]
                        header += [
                            "steps_per_trial_" + key
                            for key in num_frames_per_episode.keys()
                        ]
                        header += [
                            "num_episodes",
                            "policy_entropy",
                            "loc_entropy",
                            "loc_entropy_5",
                            "frames",
                            "FPS",
                            "duration",
                        ]
                        if not args.exp.random_action_agent:
                            header += ["return_" + key for key in return_per_episode.keys()]
                            header += ["int_reward_" + key for key in int_rewards.keys()]
                            header += ["cur_reward_" + key for key in cur_rewards.keys()]
                            header += ["values_" + key for key in values]
                            header += ["advantages_" + key for key in advantages]
                            header += [
                                "policy_loss",
                                "value_loss",
                                "grad_norm",
                                "MI_policy",
                            ]

                    data = []
                    data += [update]
                    data += num_frames_per_episode.values()
                    data += [
                        logs["num_episodes"],
                        logs["entropy"],
                        logs["loc_entropy"],
                        logs["loc_entropy_5"],
                        num_frames,
                        fps,
                        duration,
                    ]
                    if not args.exp.random_action_agent:
                        data += return_per_episode.values()
                        data += int_rewards.values()
                        data += cur_rewards.values()
                        data += values.values()
                        data += advantages.values()
                        data += [logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
                        data += [mutual_info_policy(logs["joint_dist"])]

                    if self.wandb_log:
                        wandb.log(dict(zip(header, data)))
                        wandb.log({"subroom_ids": wandb.Histogram(logs["subroom_ids"])})
                        wandb.log({"dist_travelled": logs["dist_travelled"]})

                # Do analysis

                if (
                    args.logging.analysis_interval > 0
                    and update % args.logging.analysis_interval == 0
                ):
                    if prnn_eval_bool:
                        if args.exp.onpolicy_prnn_eval:
                            analysisagent = (
                                randomagent
                                if args.exp.random_action_agent
                                else ac_agent
                            )

                            spatial_metrics = evaluate_spatial_representation(
                                predictiveNet,
                                env,
                                analysisagent,
                                sleepstd=0.03,
                                wandb_nameext="_onPolicy",
                            )
                            print(f"onPolicy sRSA={spatial_metrics['sRSA']:.4f} "
                                  f"SWdist={spatial_metrics['SWdist']:.4f}")
                            if self.wandb_log:
                                wandb.log({"SWdist_direct_onPolicy": spatial_metrics["SWdist"]})

                        if args.exp.offpolicy_prnn_eval:
                            analysisagent = (
                                ac_agent
                                if args.exp.random_action_agent
                                else randomagent
                            )

                            spatial_metrics = evaluate_spatial_representation(
                                predictiveNet,
                                env,
                                analysisagent,
                                sleepstd=0.03,
                                wandb_nameext="_offPolicy",
                            )
                            print(f"offPolicy sRSA={spatial_metrics['sRSA']:.4f} "
                                  f"SWdist={spatial_metrics['SWdist']:.4f}")
                            if self.wandb_log:
                                wandb.log({"SWdist_direct_offPolicy": spatial_metrics["SWdist"]})


                    if args.exp.analyze_agent_behav:
                        # Reuse the training rollout (free) unless the random-
                        # action path is active, which never fills the buffers.
                        opa = OnPolicyAnalysis(
                            algo,
                            timesteps=25000,
                            reuse_last_rollout=not args.exp.random_action_agent,
                        )
                        if self.wandb_log:
                            wandb.log({"MI_policy_eval": opa.mi})
                            wandb.log({"OPA_Advantages": wandb.Plotly(opa.plot_advantages())})
                            wandb.log({"OPA_Policy_Heatmaps": wandb.Plotly(opa.plot_policy_heatmaps())})
                            wandb.log({"OPA_Occupancy_Map": wandb.Plotly(opa.plot_occupancy())})

                        else:
                            RLutils.save_analysis_of_agent_behav(opa, self.model_dir, update)

                if args.logging.early_stop:
                    if (
                        return_per_episode["mean"] > 0.9
                        and return_per_episode["std"] < 0.05
                    ):
                        n_performance += 1
                        if n_performance == 25:
                            break

                # Save status

                if (
                    args.logging.save_interval > 0
                    and update % args.logging.save_interval == 0
                ):
                    status_save = {
                        StatusCkptKeys.NUM_FRAMES.value: num_frames,
                        StatusCkptKeys.UPDATE.value: update,
                    }
                    if not args.exp.random_action_agent:
                        status_save[StatusCkptKeys.MODEL_STATE.value] = acmodel.state_dict()
                        status_save[StatusCkptKeys.OPTIMIZER_STATE.value] = algo.optimizer.state_dict()

                    # if hasattr(preprocess_obss, "vocab"):
                    #     status["vocab"] = preprocess_obss.vocab.vocab

                    RLutils.save_status(status_save, self.model_dir)

                    # Save predictiveNet state if it exists and is being trained
                    if predictiveNet is not None and args.predNet.train:
                        save_pN(predictiveNet, self.model_dir + "predictiveNet_state.pt")

                    print(f"pN and ACmodel status saved at {self.model_dir}")


@hydra.main(config_path="Configs", config_name="Conf1_Adel")
def my_main(cfg: DictConfig):
    my_app(cfg)


def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ###################
    ### RUN TRAINING
    ###################

    trainer = RL_Trainer(cfg)
    trainer.run_training_loop()

    if cfg.logging.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    my_main()
