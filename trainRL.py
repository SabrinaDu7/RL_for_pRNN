import time
import datetime
import torch
import sys
import os
import shutil
import numpy as np

from omegaconf import OmegaConf, DictConfig, open_dict
import hydra
import wandb

import RLutils
from RLutils.other import device
from RLutils.model import ACModel, RecACModel, ACModelSR, ACModelTheta, ACModelThetaShared, ACModelThetaSingle
from RLutils.algo import PredictivePPOAlgo, thetaPPOalgo, SingleThetaPPOalgo
from RLutils.pc import FakePlaceCells
from RLutils.analysis import EnvironmentFeaturesAnalysis, OnPolicyAnalysis
from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.CANNNet import CANNnet
from prnn.utils.thetaRNN import LayerNormRNNCell, RNNCell
from prnn.utils.agent import RandomActionAgent

RNNoptions = {'LayerNormRNNCell' : LayerNormRNNCell ,
              'RNNCell' : RNNCell
                }


class RL_Trainer(object):

    def __init__(self, params):
 
        #############
        ## INIT
        #############

        # Get params, init WandB !!change WandB default folder
        self.params = params

        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        if params.logging.focus:
            par = eval('params.'+params.logging.focus)
            name = f"{params.exp.exp_name}_{params.logging.focus}_{par}_seed{params.exp.seed}"
        else:
            name = f"{params.exp.exp_name}_seed{params.exp.seed}"
        name_date = f"{name}_{date}/"
        self.model_name = f"{params.logging.project}/{name_date}"
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
        
        self.video_dir = RLutils.get_video_dir(self.model_name)
        RLutils.create_folders_if_necessary(self.video_dir)

        print("\n\n\nLOGGING TO: ", self.model_dir, "\n\n\n")

        self.run = wandb.init(
                                # set the wandb project where this run will be logged
                                project = params.logging.project,
                                group = self.group,
                                # group = f"{params.exp.exp_name}_width{params.rl.pc_sd}",
                                name=name,
                                id = name_date[:-1],
                                dir = self.model_dir,
                                resume='allow',
                                # track hyperparameters and run metadata
                                config = params
                                )

    def run_training_loop(self):

        args = self.params

        RLutils.seed(args.exp.seed)

        print(f"Device: {device}\n")

        # Load environment

        env = RLutils.make_env(
                               args.exp.env_name,
                               args.exp.input_type,
                               args.exp.seed + 10000,
                               self.video_dir,
                               args.logging.video_log_freq,
                               act_enc = args.predNet.action_encoding
                                )
        print("Environment loaded\n")

        # Load training status

        try:
            status = RLutils.get_status(self.model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}
        print("Training status loaded\n")

        # Load observations preprocessor

        obs_space, preprocess_obss = RLutils.get_obss_preprocessor(env.observation_space)
        # if "vocab" in status:
        #     preprocess_obss.vocab.load_vocab(status["vocab"])
        print("Observations preprocessor loaded\n")

        # Load pRNN
        if args.exp.pRNN:
            if args.logging.load_worldmodel:
                predictiveNet = PredictiveNet.loadNet(args.predNet.path)
                if not hasattr(predictiveNet.pRNN, 'hidden_size'):
                    predictiveNet.pRNN.hidden_size = predictiveNet.pRNN.rnn.cell.hidden_size
                predictiveNet.env_shell.env = env.env
                env = predictiveNet.env_shell
                print("pRNN model loaded\n")
            elif args.logging.load_acmodel:
                predictiveNet = PredictiveNet.loadNet(RLutils.get_pN(self.model_dir))
                print("pRNN model loaded\n")
            else:
                predictiveNet = PredictiveNet(env,
                                              hidden_size = args.predNet.hiddensize,
                                              pRNNtype = args.predNet.pRNNtype,
                                              learningRate = args.predNet.lr,
                                              bptttrunc = args.predNet.bptttrunc,
                                              weight_decay = args.predNet.weight_decay,
                                              neuralTimescale = args.predNet.ntimescale,
                                              dropp = args.predNet.dropout,
                                              trainNoiseMeanStd = (args.predNet.noisemean,
                                                                  args.predNet.noisestd),
                                              f = args.predNet.sparsity)
                print("pRNN model initialized\n")
            args.predNet.hiddensize = predictiveNet.hidden_size
            # predictiveNet.pRNN.to(device)
            predictiveNet.env_shell.hd_trans = np.array([-1,1,0,0]) # TODO: remove later
        else:
            predictiveNet = None

        # Load models
        if args.exp.recurrence-1:
            acmodel = RecACModel(obs_space,
                                 env.action_space,
                                 RNNoptions[args.predNet.cell],
                                 args.predNet.hiddensize,
                                 args.exp.with_obs,
                                 args.exp.rgb,
                                 args.exp.with_HD)
            
        elif args.exp.theta:
            if args.exp.single_theta:
                acmodel = ACModelThetaSingle(obs_space, env.action_space,
                                             args.predNet.hiddensize,
                                             predictiveNet.pRNN.k, args.rl.value_type)
            elif args.exp.shared_weights:
                acmodel = ACModelThetaShared(obs_space, env.action_space,
                                             args.predNet.hiddensize,
                                             predictiveNet.pRNN.k, args.rl.value_type)
            else:
                acmodel = ACModelTheta(obs_space, env.action_space,
                                    args.predNet.hiddensize, args.exp.with_obs,
                                    args.exp.rgb, predictiveNet.pRNN.k, args.rl.value_type)
            
        elif args.exp.PC or args.exp.CANN or args.exp.pRNN:
            acmodel = ACModelSR(obs_space, env.action_space,
                                args.predNet.hiddensize, args.exp.with_obs,
                                args.exp.rgb)

        else:
            acmodel = ACModel(obs_space, env.action_space, args.exp.with_HD,
                              args.exp.rgb)

        if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
            print("Existing model found")
        acmodel.to(device)
        print("AC model loaded\n")

        # Load place cells
        if args.exp.PC:
            PC = FakePlaceCells(env, args.predNet.hiddensize, args.rl.pc_sd, args.exp.seed)
        else:
            PC = None

        if args.exp.CANN:
            CANN = CANNnet(env,
                           hidden_size = args.predNet.hiddensize,
                           mapsize = [env.width, env.height])
        else:
            CANN = None
            

        # Load algo
        pastSR = not('prevAct' in str(predictiveNet.pRNN))
        if args.exp.single_theta:
            algo = SingleThetaPPOalgo(
                                env, acmodel, predictiveNet, device, args.rl.frames, args.rl.discount,
                                args.rl.lr, args.rl.gae_lambda, args.rl.entropy_coef, args.rl.value_loss_coef,
                                args.rl.max_grad_norm, args.exp.recurrence, args.rl.optim_eps, args.rl.ppo_clip_eps,
                                args.rl.ppo_epochs, args.rl.ppo_batch_size, preprocess_obss, PC, CANN,
                                args.predNet.train, args.predNet.noisemean, args.predNet.noisestd, args.exp.intrinsic,
                                args.rl.k_int, pastSR, args.rl.eval_type, args.rl.value_type
                                )
        
        elif args.exp.theta:
            algo = thetaPPOalgo(
                                env, acmodel, predictiveNet, device, args.rl.frames, args.rl.discount,
                                args.rl.lr, args.rl.gae_lambda, args.rl.entropy_coef, args.rl.value_loss_coef,
                                args.rl.max_grad_norm, args.exp.recurrence, args.rl.optim_eps, args.rl.ppo_clip_eps,
                                args.rl.ppo_epochs, args.rl.ppo_batch_size, preprocess_obss, PC, CANN,
                                args.predNet.train, args.predNet.noisemean, args.predNet.noisestd, args.exp.intrinsic,
                                args.rl.k_int, pastSR, args.rl.eval_type, args.rl.value_type
                                )
        else:
            algo = PredictivePPOAlgo(
                                     env, acmodel, predictiveNet, device, args.rl.frames, args.rl.discount,
                                     args.rl.lr, args.rl.gae_lambda, args.rl.entropy_coef, args.rl.value_loss_coef,
                                     args.rl.max_grad_norm, args.exp.recurrence, args.rl.optim_eps, args.rl.ppo_clip_eps,
                                     args.rl.ppo_epochs, args.rl.ppo_batch_size, preprocess_obss, PC, CANN,
                                     args.predNet.train, args.predNet.noisemean, args.predNet.noisestd, args.exp.intrinsic,
                                     args.rl.k_int, pastSR
                                     )


        if "optimizer_state" in status:
            algo.optimizer.load_state_dict(status["optimizer_state"])
        print("Optimizer loaded\n")


        # Create random agent for analysis

        action_probability = np.array([0.15,0.15,0.6,0.1])
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
            exps, logs1 = algo.collect_experiences()
            logs2 = algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % args.logging.log_interval == 0:
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = RLutils.synthesize(logs["return_per_episode"], signs=True)
                num_frames_per_episode = RLutils.synthesize(logs["num_frames_per_episode"])
                int_rewards = RLutils.synthesize(logs["intrinsic_rewards"], abs=True)

                if not header:
                    header = ["return_" + key for key in return_per_episode.keys()]
                    header += ["int_reward_" + key for key in int_rewards.keys()]
                    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                    header += ["entropy", "value", "policy_loss",
                               "value_loss", "grad_norm",
                               "loc_entropy", "loc_entropy_5", "projection similarity"]
                    header += ["frames", "FPS", "duration", "episodes"]

                data = []
                data += return_per_episode.values()
                data += int_rewards.values()
                data += num_frames_per_episode.values()
                data += [logs["entropy"], logs["value"], logs["policy_loss"],
                         logs["value_loss"], logs["grad_norm"],
                         logs["loc_entropy"], logs["loc_entropy_5"],
                         logs["proj_sim"]]
                data += [num_frames, fps, duration, logs["num_episodes"]]

                wandb.log(dict(zip(header, data)))

            # Do analysis

            if args.logging.analysis_interval > 0 and update % args.logging.analysis_interval == 0:
                EFS = EnvironmentFeaturesAnalysis(env, randomagent, acmodel, predictiveNet, 20000)
                if not error_map:
                    error_map = EFS.error_map(HDs=False)
                    error_map.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                    paper_bgcolor='rgba(0, 0, 0, 0)')
                    error_map.write_image(self.model_dir+"/"+str(update)+"_errors.png")

                fig = EFS.policy_map()
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                  paper_bgcolor='rgba(0, 0, 0, 0)')
                fig.write_image(self.model_dir+"/"+str(update)+"_policy.png")
                fig = EFS.values_map(HDs=False)
                fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                                  paper_bgcolor='rgba(0, 0, 0, 0)')
                fig.write_image(self.model_dir+"/"+str(update)+"_values.png")

                # OPA = OnPolicyAnalysis(algo, 20000)
                # fig = OPA.plot_advantages()
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_advantages.png")
                # fig = OPA.plot_deltas()
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_deltas_true.png")
                # fig = OPA.plot_deltas(zmin=-0.2)
                # fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                #                   paper_bgcolor='rgba(0, 0, 0, 0)')
                # fig.write_image(self.model_dir+"/"+str(update)+"_deltas.png")

            if args.logging.early_stop:
                if return_per_episode['mean']>0.9 and return_per_episode['std']<0.05:
                    n_performance += 1
                    if n_performance == 25:
                        break

            # Save status

            if args.logging.save_interval > 0 and update % args.logging.save_interval == 0:
                status = {"num_frames": num_frames, "update": update,
                        "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
                # if hasattr(preprocess_obss, "vocab"):
                #     status["vocab"] = preprocess_obss.vocab.vocab
                RLutils.save_status(status, self.model_dir)
                print("Status saved")

@hydra.main(config_path="Configs", config_name="Conf1")
def my_main(cfg: DictConfig):
    my_app(cfg)

def my_app(cfg: DictConfig): 
    print(OmegaConf.to_yaml(cfg))

    # Add an environment variable for storage
    os.environ['RL_STORAGE'] = cfg.logging.logdir

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