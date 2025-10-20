import warnings

### Suppress warnings ###
warnings.filterwarnings("ignore", category=UserWarning)

import time
import datetime
import os
import numpy as np
import torch.nn as nn
from typing import Mapping, Any

from omegaconf import OmegaConf, DictConfig
import hydra
import wandb

import RLutils
from RLutils.other import DEVICE
from RLutils.model import (
    ACModel,
    RecACModel,
    ACModelSR,
    ACModelTheta,
    ACModelThetaShared,
    ACModelThetaSingle,
)
from RLutils.agent import ActorCriticAgent
from RLutils.algo import PredictivePPOAlgo, thetaPPOalgo, SingleThetaPPOalgo
from RLutils.pc import FakePlaceCells
from RLutils.analysis import (
    EnvironmentFeaturesAnalysis,
    OnPolicyAnalysis,
    mutual_info_policy,
)
from prnn.utils import (
    PredictiveNet,
    CANNnet,
    LayerNormRNNCell,
    RNNCell,
    RandomActionAgent,
    load_pN, 
    save_pN, 
    CkptKeys
)
import utils.dev_env as dev
PRNN_CKPT, ACMODEL_STATUS_CKPT = dev.get_ckpt_env_vars()
WANDB_ENTITY, WANDB_PROJECT = dev.get_wandb_env_vars()

RNNoptions = {"LayerNormRNNCell": LayerNormRNNCell, "RNNCell": RNNCell}

class RL_Trainer(object):
    def __init__(self, params):
        #############
        ## INIT
        #############

        # Get params, init WandB !!change WandB default folder
        self.params = params

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        if params.logging.focus:
            par = eval("params." + params.logging.focus)
            name = f"{params.exp.exp_name}_{params.logging.focus}_{par}_seed{params.exp.seed}"
        else:
            name = f"{params.exp.exp_name}_seed{params.exp.seed}"
        name_date = f"{name}_{date}/"
        self.model_name = f"{WANDB_PROJECT}/{name_date}"
        if params.logging.focus:
            self.group = f"{params.exp.exp_name}_{params.logging.focus}_{par}"
        else:
            self.group = params.exp.exp_name

        # Set run dir

        if params.logging.load_acmodel:
            self.model_dir = params.logging.load_acmodel
        else:
            self.model_dir = RLutils.get_model_dir(self.model_name)
            RLutils.create_folders_if_necessary(self.model_dir)

        self.video_dir = (
            RLutils.get_video_dir(self.model_name)
            if params.logging.video_log_freq != 0
            else ""
        )
        RLutils.create_folders_if_necessary(self.video_dir)

        print("\n\n\nLOGGING TO: ", self.model_dir, "\n\n\n")

        self.run = wandb.init(
            # set the wandb project where this run will be logged
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=self.group,
            # group = f"{params.exp.exp_name}_width{params.rl.pc_sd}",
            name=name,
            id=name_date[:-1],
            dir=self.model_dir,
            resume="allow",
            # track hyperparameters and run metadata
            config=OmegaConf.to_container(params, resolve=True),
        )

    def run_training_loop(self):
        args = self.params

        RLutils.seed(args.exp.seed)

        print(f"Device: {DEVICE}\n")

        # Load environment

        env = RLutils.make_env(
            args.exp.env_name,
            args.exp.input_type,
            args.exp.seed + 10000,
            self.video_dir,
            args.logging.video_log_freq,
            act_enc=args.predNet.action_encoding,
        )
        print("Environment loaded\n")

        # Load training status

        if args.logging.load_acmodel:
            status = RLutils.get_status(ACMODEL_STATUS_CKPT) 
        else:
            status = {"num_frames": 0, "update": 0}
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
            wandb_log=True,
        )

        args.predNet.hiddensize = predictiveNet.hidden_size
        # predictiveNet.pRNN.to(device)
        predictiveNet.env_shell.hd_trans = np.array([-1, 1, 0, 0])  # TODO: remove later

        # Try to load predictiveNet state from new format first
        if args.logging.load_worldmodel:
            load_pN(predictiveNet, PRNN_CKPT)
            print("\n" + "=" * 10)
            print(f"Existing pRNN model found at {PRNN_CKPT} and loaded from state dict")
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

        if "model_state" in status:
            print("\n" + "=" * 10)
            state_dict: Mapping[str, Any] = status["model_state"]
            acmodel.load_state_dict(state_dict)
            print("Existing model found")
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
            )

        if "optimizer_state" in status:
            optimizer_state: dict[str, Any] = status["optimizer_state"]
            algo.optimizer.load_state_dict(optimizer_state)
        print("Optimizer loaded\n")

        # Create random agent for analysis

        action_probability = np.array([0.15, 0.15, 0.6, 0.1])
        randomagent = RandomActionAgent(env.action_space, action_probability)

        # Train model
        num_frames = status["num_frames"]
        update = status["update"]
        start_time = time.time()
        header = False

        n_performance = 0
        error_map = None

        while num_frames < args.rl.steps:
            # Update model parameters
            update_start_time = time.time()

            if args.exp.random_action_agent:
                logs = algo.randomAgent_collect_exp_and_update(randomagent)
            else:
                exps, logs1 = algo.collect_experiences()
                logs2 = algo.update_parameters(
                    exps, update_params=(not args.exp.random_init_control)
                )
                logs = {**logs1, **logs2}

            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.logging.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                num_frames_per_episode = RLutils.synthesize(
                    logs["num_frames_per_episode"]
                )

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

                wandb.log(dict(zip(header, data)))

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
                            else ActorCriticAgent(
                                env.action_space, acmodel, predictiveNet, DEVICE
                            )
                        )

                        _, _, _ = predictiveNet.calculateSpatialRepresentation(
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
                            ActorCriticAgent(
                                env.action_space, acmodel, predictiveNet, DEVICE
                            )
                            if args.exp.random_action_agent
                            else randomagent
                        )

                        _, _, _ = predictiveNet.calculateSpatialRepresentation(
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
                    wandb.log({"MI_policy_eval": opa.mi})
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
                    "num_frames": num_frames,
                    "update": update,
                }
                if not args.exp.random_action_agent:
                    status_save["model_state"] = acmodel.state_dict()
                    status_save["optimizer_state"] = algo.optimizer.state_dict()
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

    # Add an environment variable for storage
    os.environ["RL_STORAGE"] = cfg.logging.logdir

    ###################
    ### RUN TRAINING
    ###################

    trainer = RL_Trainer(cfg)
    try:
        trainer.run_training_loop()
    finally:
        wandb.finish()


if __name__ == "__main__":
    my_main()
