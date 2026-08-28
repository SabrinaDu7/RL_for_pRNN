2026-08-28 · branch `sdu/predict-next-obs`

# What the pRNN and the policy actually receive, and what they are asked for

**Goal.** Before changing the world model's prediction target, establish exactly what
each learner is given and what it is scored against, at which timestep, with the
evidence for every claim. The change we want — *the pRNN predicts the observation that
results from the action* — turns out to be reachable by three different mechanisms that
are NOT equivalent, and the wrong one is already half-wired into the codebase under a
name that suggests it is the right one.

Everything below is **confirmed** by reading the cited line unless marked inferred.
Line numbers are `curious_george/` unless prefixed `prnn:` (the pinned package, at
`.venv/lib/python3.10/site-packages/prnn/`).

## Notation

`obs[t]` is the observation the agent sees at step `t`. `a[t]` is the action it takes
after seeing `obs[t]`; taking it produces `obs[t+1]`. `h[t]` is the hidden state.

## 1. The current wiring

```
                obs[0]      obs[1]      obs[2]      obs[3] ...
                  |           |           |
                  v           v           v
   pRNN in    ( obs[0],a[0] ) ( ---- ,a[1] ) ( ---- ,a[2] )      <- inMask zeroes obs
   target        obs[0]      obs[1]      obs[2]                  <- predOffset = 0
   MSE           e[0]        e[1]        e[2]
   policy r      e[1]        e[2]        e[3]                    <- next_obs shifts by 1
```

| what | value | where |
| --- | --- | --- |
| architecture | `thRNN_5win` = `partial(MaskedRNN, cell=LayerNormRNNCell, k=5)` | prnn:`Architectures.py:858` |
| the class | `class MaskedRNN(pRNN)` - extends `pRNN` directly, NOT `pRNN_th` | prnn:`Architectures.py:698` |
| prediction offset | `predOffset=0`, hard-coded in the `super()` call | prnn:`Architectures.py:756` |
| the slice it drives | `obs_target = obs_in[:, self.predOffset:, :]` in `pRNN.restructure_inputs` | prnn:`Architectures.py:203` |
| action offset | `actOffset=0`, a real constructor parameter | prnn:`Architectures.py:718` |
| observation masking | `inMask = np.full(k+1, False); inMask[0] = True` -> 1 shown in 6 | prnn:`Architectures.py:739-740` |
| ...applied | `obs_out = obs_in * self._tile_mask(self.inMask_f, ...)` | prnn:`Architectures.py:235` |
| reward shift | `REWARD_ALIGNMENTS = {"legacy": 0, "next_obs": 1}` | `rl/update/rewards.py:25` |
| in use | `reward_alignment: RewardAlignment = RewardAlignment.NEXT_OBS` | `configs.py:509` |

`models/prnn_adapter.py:8-13` already states this and flags the trap:

> All `*_5win` architectures set `predOffset=0`, so `predict()` returns `obs_pred[t]`
> targeting `obs[t]` (the SAME timestep) [...] Docstrings in prnn claiming t+1 describe
> the base-class default (`predOffset=1`) which every 5win subclass overrides.

**Independent confirmation from data, not from code.** The published rollout artifact
(*Inside the masked pRNN, one timestep at a time*, 14 steps of a random-action rollout
through a trained checkpoint) renders the "given to pRNN" and "target" cells at `t = 0`
from **byte-identical base64 strings**, and labels every curiosity-reward cell
`<- MSE at t = <t+1>`.

### The consequence, stated plainly

`a[t]` enters the hidden state at the same step whose target is `obs[t]` — an
observation that already existed before `a[t]` was chosen. **The action cannot inform
the prediction it is paired with.** `reward_alignment="next_obs"` then shifts the reward
by one so the policy is at least credited for the error its action did cause; the module
docstring (`rl/update/rewards.py:9-17`) says exactly this about the `legacy` default it
replaced — *"crediting the action with surprise it did not cause."*

So the reward shift is a correction applied downstream of a misalignment that is still
present upstream in the world model's own objective.

### A second consequence, not previously written down

On phase 0 the observation input is NOT masked, and the target is `obs[t]` — **the
target is present in the input**. That step is partly an autoencoding problem rather
than a prediction problem. The artifact's own numbers are consistent with this: the two
phase-0 steps score MSE `0.00285` and `0.00553`, against `0.00575`-`0.01715` across the
ten masked steps — at or below the bottom of the masked range. Suggestive, not decisive:
`n = 2` phase-0 steps in one rollout. **Inferred**, and cheap to settle by scoring MSE
grouped by phase over a full probe.

## 2. Three mechanisms, and only one is what we asked for

| | at step `t`, inputs | target | what it asks |
| --- | --- | --- | --- |
| **now** `predOffset=0` | `obs[t]` (masked), `a[t]` | `obs[t]` | reconstruct what you just saw |
| **A** `predOffset=1` | `obs[t]` (masked), `a[t]` | `obs[t+1]` | **anticipate what the action produces** |
| **B** `actOffset=1` | `obs[t]` (masked), `a[t-1]` | `obs[t]` | explain what the last action produced |

**A is the change you described.** The target moves forward; the action and the
observation it causes land in the same timestep; and the target is never in the input,
so the phase-0 degeneracy above disappears entirely.

**B already exists and is misleadingly named.** `thRNN_5win_prevAct = partial(MaskedRNN,
..., actOffset=1)` (prnn:`Architectures.py:877`), exposed as
`pRNNtypes.masked_nextstep` (prnn:`enums.py:21`). The enum name says "nextstep"; the
mechanism shifts the *action* backward, not the target forward — `actOffset`'s own
docstring is *"Number of timesteps to offset actions by (backwards)"*
(prnn:`Architectures.py:734`; the base class says the same at prnn:`Architectures.py:72`),
implemented as a front-pad then a tail-drop at prnn:`Architectures.py:86-87,199-200`.

🔴 **B is not A re-indexed.** Re-index B at `t' = t-1` and its inputs become
`(obs[t'+1], a[t'])` against target `obs[t'+1]` — on unmasked phases the target is again
sitting in the input. B keeps the degeneracy; A removes it. A name that predicts
"nextstep" and delivers a backward action shift is exactly the kind of misleading name
this repo treats as a priority defect, and it is in the pinned dependency, not here.

## 3. What is coupled to the architecture choice

Switching architecture is not a one-field change. Four things move together:

| coupling | where |
| --- | --- |
| `pastSR = "prevAct" not in predictiveNet.pRNNtype` | `training/setup.py:236` |
| `assert pastSR ^ ("Next" in str(self.env.encodeAction))` — forces `SpeedNextHD` | `rl/algo.py:184` |
| `assert pastSR is not ("prevAct" in prnn.pRNNtype)` | `rl/collect/agent.py:24` |
| the policy's observation: `state.obs_b[0] if algo.pastSR else last_post_obs` | `rl/algo.py:78` |

And the entry point is currently closed: `ArchPrnnCfg.prnn_type` is a **property that
always returns `pRNNtypes.masked`** (`configs.py:357-362`), documented as *"Fixed. The
prevAct variant is retired."*

**This matters for route A specifically: none of that coupling applies.** `pastSR` keys
on the string `"prevAct"`, and route A changes `predOffset`, not the action indexing. A
`predOffset=1` masked architecture keeps `pastSR=True`, keeps `SpeedHD`, keeps the
policy's observation source, and trips none of the three asserts. **Route A is the
smaller change as well as the one asked for.**

## 4. What route A actually costs

- 🔴 **It cannot be done with a `partial`, and I initially thought it could.**
  `MaskedRNN.__init__` does NOT take `predOffset` (signature at
  prnn:`Architectures.py:707-721`); it hard-codes `predOffset=0` in its `super()` call
  at prnn:`Architectures.py:756`. `partial(MaskedRNN, predOffset=1)` routes it through
  `**cell_kwargs` into the same `super()` call and raises *multiple values for keyword
  argument 'predOffset'* - loudly, which is the one good thing here. Route A therefore
  needs `MaskedRNN`'s signature to accept and forward `predOffset`, i.e. an edit to the
  **pinned `prnn` package** and a re-pin.
- **`NextStepRNN` already does expose it** (prnn:`Architectures.py:642`, default
  `predOffset=1` at prnn:`Architectures.py:660`) and is used by the `Autoencoder*Pred`
  partials at prnn:`Architectures.py:832-837`. It is a different family - no `inMask` -
  so it is prior art for the parameter, not a drop-in.
- **The sequence shortens by one step.** `clip_mask` takes
  `minsize = min(obs_in.size(1), act.size(1), obs_target.size(1))`
  (prnn:`Architectures.py:226`), and `obs_target` loses its first row under
  `predOffset=1`, so the last action of each segment is dropped. `rl/update/rewards.py:14-17`
  records that the adapter already handles the analogous case for `next_obs` by
  *"extend[ing] the per-episode predict pass by one zero-action step"* — the same
  treatment is what keeps every action rewarded here.
- **`reward_alignment` becomes redundant.** Under A the MSE at `t` is already the error
  on the observation `a[t]` produced, so the correct setting becomes `legacy` (offset 0)
  — the *opposite* of today. Getting this backwards would double-shift the reward, and
  nothing would fail loudly.

## 5. Open decisions, for Sabrina

1. **Where does the new variant live?** A `predOffset=1` partial belongs in `prnn`
   beside its siblings, which means a fork edit and a re-pin. The alternative — passing
   `predOffset` through `PredictiveNet` from our config — avoids the fork but puts an
   architecture parameter in two homes.
2. **`prnn_type` is a hard-coded property.** Route A needs it to become a real field
   again. Reopening it also reopens `masked_nextstep`, which section 2 argues is
   misnamed; leaving that name in place while adding a genuine next-step variant would
   put two contradictory meanings of "nextstep" in one enum.
3. **`reward_alignment` must flip to `legacy` in the same commit**, per section 4.

## 6. What would confirm this document

Not yet run:

- **MSE grouped by phase** over a probe, to settle whether phase 0 is measurably easier
  than phases 1-5 (section 1, marked inferred).
- **A shape assertion** that under `predOffset=1`, `predict()`'s output at `t` is scored
  against `obs[t+1]` — the direct analogue of the artifact, on the new path.
