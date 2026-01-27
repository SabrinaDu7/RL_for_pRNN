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
import RLutils
from RLutils import (
    DEVICE,
    ACModel,
    RecACModel,
    ACModelSR,
    ACModelTheta,
    ACModelThetaShared,
    ACModelThetaSingle,
    ActorCriticAgent,
    PredictivePPOAlgo, 
    thetaPPOalgo, 
    SingleThetaPPOalgo,
    FakePlaceCells,
    OnPolicyAnalysis,
    mutual_info_policy,
    get_agent,
)

from prnn.utils import (
    PredictiveNet,
    CANNnet,
    LayerNormRNNCell,
    RNNCell,
    RandomActionAgent,
    load_pN, 
    save_pN, 
)
from utils import load_statedict_from_acmodel_status

WANDB_ENTITY, WANDB_PROJECT = get_wandb_env_vars(omt=False)
RNNoptions = {"LayerNormRNNCell": LayerNormRNNCell, "RNNCell": RNNCell}

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
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                group=self.group,
                name=run_name,
                id=run_name,
                dir=self.model_dir,
                resume="allow",
                config=OmegaConf.to_container(params, resolve=True),
            )

    def run_training_loop(self):
        args = self.params

        RLutils.seed(args.exp.seed)

        print(f"Device: {DEVICE}\n")

        # Load environment (default size = 16)
        agent_start_pos = None
        agent_start_dir = None

        if 'FourRooms' in args.exp.env_name and not args.exp.start_rand:
            agent_start_pos=(1, 1)
            agent_start_dir=1

        env = RLutils.make_env(
            env_key=args.exp.env_name,
            input_type=args.exp.input_type,
            seed=args.exp.seed + 10000,
            act_enc=args.predNet.action_encoding,
            open_all_paths=False, # Only applicable for FourRooms env
            subroom_size=args.exp.env_subroom_size, # Only applicable for FourRooms env
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
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
        if args.exp.recurrence - 1:
            acmodel = RecACModel(
                obs_space,
                env.action_space,
                RNNoptions[args.predNet.cell],
                args.predNet.hiddensize,
                args.exp.with_obs,
                args.exp.rgb,
                args.exp.with_HD,
            )

        elif args.exp.theta:
            if args.exp.single_theta:
                acmodel = ACModelThetaSingle(
                    obs_space,
                    env.action_space,
                    args.predNet.hiddensize,
                    predictiveNet.pRNN.k,
                    args.rl.value_type,
                )
            elif args.exp.shared_weights:
                acmodel = ACModelThetaShared(
                    obs_space,
                    env.action_space,
                    args.predNet.hiddensize,
                    predictiveNet.pRNN.k,
                    args.rl.value_type,
                )
            else:
                acmodel = ACModelTheta(
                    obs_space,
                    env.action_space,
                    args.predNet.hiddensize,
                    args.exp.with_obs,
                    args.exp.rgb,
                    predictiveNet.pRNN.k,
                    args.rl.value_type,
                )

        elif args.exp.PC or args.exp.CANN or args.exp.pRNN:
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

        # Load place cells
        if args.exp.PC:
            PC = FakePlaceCells(
                env, args.predNet.hiddensize, args.rl.pc_sd, args.exp.seed
            )
        else:
            PC = None

        if args.exp.CANN:
            CANN = CANNnet(
                env,
                hidden_size=args.predNet.hiddensize,
                mapsize=[env.width, env.height],
            )
        else:
            CANN = None

        if args.predNet.train:
            assert args.predNet.seqdur > 0, "Set an appropriate seqdur"
        else:
            args.predNet.seqdur = 0

        # Load algo
        pastSR = not ("prevAct" in str(predictiveNet.pRNN))
        print("pastSR:", pastSR)
        if args.exp.single_theta:
            algo = SingleThetaPPOalgo(
                env,
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
                args.exp.recurrence,
                args.rl.optim_eps,
                args.rl.ppo_clip_eps,
                args.rl.ppo_epochs,
                args.rl.ppo_batch_size,
                preprocess_obss,
                PC,
                CANN,
                args.predNet.train,
                args.predNet.noisemean,
                args.predNet.noisestd,
                args.exp.intrinsic,
                args.rl.k_int,
                pastSR,
                args.rl.eval_type,
                args.rl.value_type,
            )

        elif args.exp.theta:
            algo = thetaPPOalgo(
                env,
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
                args.exp.recurrence,
                args.rl.optim_eps,
                args.rl.ppo_clip_eps,
                args.rl.ppo_epochs,
                args.rl.ppo_batch_size,
                preprocess_obss,
                PC,
                CANN,
                args.predNet.train,
                args.predNet.noisemean,
                args.predNet.noisestd,
                args.exp.intrinsic,
                args.rl.k_int,
                pastSR,
                args.rl.eval_type,
                args.rl.value_type,
            )
        else:
            algo = PredictivePPOAlgo(
                env,
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
                args.exp.recurrence,
                args.rl.optim_eps,
                args.rl.ppo_clip_eps,
                args.rl.ppo_epochs,
                args.rl.ppo_batch_size,
                preprocess_obss,
                PC,
                CANN,
                args.predNet.train,
                args.predNet.noisemean,
                args.predNet.noisestd,
                args.predNet.seqdur,
                args.exp.intrinsic,
                args.rl.k_int,
                pastSR,
                args.exp.curious_agent,
                args.rl.k_curious,
                args.rl.use_rnd,
                args.rl.rnd_output_dim,
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

                # Print logs

                if update % args.logging.log_interval == 0:
                    fps = logs["num_frames"] / (update_end_time - update_start_time)
                    duration = int(time.time() - start_time)
                    num_frames_per_episode = RLutils.synthesize(
                        logs["num_frames_per_episode"]
                    )

                    predictiveNet.plotSampleTrajectory(
                            env=env,
                            agent=ac_agent,
                        ) # Logs to wandb inside the function if predictiveNet.wandb_log is True
                    predictiveNet.pRNN.to(DEVICE) # LANDMINE: plotSampleTraj calls agent's getObs, which for some reason sets pRNN to cpu!!!

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
                    if prnn_eval_bool and not args.exp.CANN:
                        predictiveNet.pRNN.to("cpu")

                        if args.exp.onpolicy_prnn_eval:
                            analysisagent = (
                                randomagent
                                if args.exp.random_action_agent
                                else ac_agent
                            )

                            _, _, _, sRSA = predictiveNet.calculateSpatialRepresentation(
                                env,
                                analysisagent,
                                trainDecoder=True,
                                trainHDDecoder=False,
                                saveTrainingData=False,
                                bitsec=False,
                                calculatesRSA=True,
                                sleepstd=0.03,
                                wandb_nameext="_onPolicy",
                            )

                        if args.exp.offpolicy_prnn_eval:
                            analysisagent = (
                                ac_agent
                                if args.exp.random_action_agent
                                else randomagent
                            )

                            _, _, _, sRSA = predictiveNet.calculateSpatialRepresentation(
                                env,
                                analysisagent,
                                trainDecoder=True,
                                trainHDDecoder=False,
                                saveTrainingData=False,
                                bitsec=False,
                                calculatesRSA=True,
                                sleepstd=0.03,
                                wandb_nameext="_offPolicy",
                            )

                        predictiveNet.pRNN.to(DEVICE)
                        
                    if args.exp.analyze_agent_behav:
                        opa = OnPolicyAnalysis(algo, timesteps=25000)
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
