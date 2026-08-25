# How the pRNN and action network are updated
For pastSR = True and rewards = next_obs

## Episode start
1. obs_0 = env.reset(); pRNN state reset: h ← randInit noise, φ ← 0; and the policy's SR is initialized to zeros: SR_0 = 0 (not h — the very first action is chosen with a blank SR).

## One rollout step t (repeats for the whole episode)
2. Action network forward: dist, value_t = ACModelSR(obs_t, SR=SR_t) — inputs are the current observation's HD (one-hot) and SR_t; no gradient here.
3. Action: a_t ~ dist; store log_prob_t, value_t, SR_t, mask_t.
4. Environment: obs_{t+1}, r_t, done_t = env.step(a_t).
5. pRNN hidden-state update (the pastSR-defining step): encode x_t = [obs_t · inMask[φ], SpeedHD(a_t, HD_t) · actMask[φ]] — the pre-action observation and the action just taken; then h ← cell(h, x_t), φ ← (φ+1) mod 6, and SR_{t+1} = h. Note obs_{t+1} does not enter the pRNN at this step — it feeds step t+1's update. The SR the policy will use next therefore encodes the agent's history up to and including leaving position t, i.e. it lags one position ("past" SR).
6. If done_t (natural, or forced every seqdur steps): record the episode boundary and last_obs = obs_{t+1}; reset h ← randInit noise, φ ← 0, SR ← 0; env.reset(). Otherwise loop to step 2 with obs_{t+1}, SR_{t+1}.

## After the rollout (2048 steps collected) — reward computation, retrospective, per episode
7. For each episode segment [obs_0..obs_{L-1}] + [last_obs, last_obs] with actions [a_0..a_{L-1}, 0] (the appended step's encoded action row is zeroed — the init_sr convention), run one fresh predict() pass (its own random-init state; independent of the rollout's h). Prediction row i targets obs_i, with input (obs_i · inMask, a_i) on the same 1-in-6 mask schedule.
8. next_obs curiosity reward: curious_r_t = MSE(row t+1) = ‖pred(obs_{t+1}) − obs_{t+1}‖² — the error on the observation a_t produced. Causally, row t+1's prediction is driven by the recurrent state that already ingested (obs_t, a_t), plus its own (5/6-masked) input. The final action a_{L-1} gets the appended zero-action row targeting last_obs — same formula as every other step.
9. Total reward: R_t = r_t + k_curious · curious_r_t (extrinsic r_t is 0 in LRoom), then GAE runs backward over each env stream using the stored value_t and masks → advantage_t, return_t = value_t + advantage_t.

## Parameter updates
10. Action network (PPO): 4 epochs of shuffled minibatches; re-forward ACModelSR(obs_t, SR=SR_t) on the stored SRs (the pRNN is not backpropped through — SR is a frozen input), ppo_clip loss on (log_prob_t, advantage_t, return_t) → Adam step on the AC weights only.
11. pRNN training: per episode segment, one trainStep(obs_0..last_obs, a_0..a_{L-1}) — the same prediction objective as step 7, but with gradients → the pRNN's own optimizer updates its weights.

The two coupling loops this creates: the pRNN shapes the policy twice — its hidden state is the policy's spatial input (steps 2, 5), and its prediction error is the policy's reward (steps 8–9). The policy shapes the pRNN once — it chooses the data distribution the pRNN trains on (step 11). Note the deliberate asymmetry between steps 5 and 8: the online hidden-state update never sees obs_{t+1} at step t, while the reward for a_t is exactly about obs_{t+1} — that's what next_obs fixed relative to legacy (which scored obs_t, an observation the action couldn't have influenced).
