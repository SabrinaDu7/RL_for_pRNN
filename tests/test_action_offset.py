"""`arch_prnn.action_offset` is the circuit, and these are the properties it has.

offset 0: row t = (obs[t], a[t])   - the action chosen AFTER seeing obs[t]
offset 1: row t = (obs[t], a[t-1]) - the action that PRODUCED obs[t]

Everything else is meant to be identical: same architecture, same encoding, same
`predOffset`, same upstream `actOffset=0`. These tests pin that, and pin the two
things the shift is built by hand to get right - `HD[0]` in row 0, and the
segment's last action surviving instead of being clipped away.

Zero noise and zero dropout throughout, so a difference is the circuit and not a
draw.
"""

import numpy as np
import pytest
import torch

from prnn.utils import ActionEncodingsEnum, MinigridEnvNames, PredictiveNet

from curious_george import AgentInputType, make_env
from curious_george.models.policy import ACModelSR
from curious_george.models.prnn_adapter import FORWARD_IDX, PRNNAdapter
from curious_george.rl.algo import PredictivePPOAlgo
from curious_george.rl.collect.format import get_obss_preprocessor

SEED = 11
L = 10
HIDDEN = 32


def _env(seed=SEED):
    return make_env(
        env_key=MinigridEnvNames.LRoom,
        input_type=AgentInputType.H_PO.value,
        act_enc=ActionEncodingsEnum.SpeedHD.value,
        seed=seed,
    )


def _net(env):
    pN = PredictiveNet(
        env, hidden_size=HIDDEN, pRNNtype="thRNN_5win",
        trainNoiseMeanStd=(0, 0), dropp=0.0, wandb_log=False,
    )
    pN.pRNN.eval()
    return pN


@pytest.fixture(scope="module")
def segment():
    """One fixed trajectory: observations, actions, and the final observation."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = _env()
    rng = np.random.default_rng(SEED)
    obs = env.reset()
    obss, acts = [], []
    for _ in range(L):
        a = int(rng.choice(4, p=[0.2, 0.2, 0.5, 0.1]))
        obss.append(obs)
        acts.append(a)
        obs = env.step(np.array([a]))[0]
    return obss, np.array(acts), obs


def _rows(offset, segment):
    """The (observation, action, target) tensors the pRNN is handed."""
    obss, acts, last = segment
    env = _env()
    pN = _net(env)
    ad = PRNNAdapter(pN, torch.device("cpu"), pastSR=offset == 0)
    obs_f, act_f = ad.seq2pred(list(obss) + [last], acts)
    with torch.no_grad():
        x, target, _ = pN.pRNN.restructure_inputs(obs_in=obs_f, act=act_f)
    n = pN.obs_size
    return x[0, :, :n], x[0, :, n:].to(torch.int64), target[0], ad


def test_only_the_forward_bit_moves(segment):
    """The action shifts back one row; the head direction does not.

    Asserting the whole vector shifted would FAIL, and that failure is the
    design: `SpeedHD` packs a speed and a head direction into one row, and only
    the speed belongs to the previous step. HD[t] is which way you were facing
    when you saw obs[t], so it stays put.
    """
    obs0, act0, tgt0, ad = _rows(0, segment)
    obs1, act1, tgt1, _ = _rows(1, segment)
    A = ad.num_acts
    n = min(len(act0), len(act1))

    assert torch.equal(obs0[:n], obs1[:n]), "the observation input must not move"
    assert torch.equal(tgt0[:n], tgt1[:n]), "the target must not move"
    assert (act1[0, :A] == 0).all(), "row 0 has no preceding action to carry"
    for t in range(1, n):
        assert torch.equal(act1[t, :A], act0[t - 1, :A]), f"forward bit, row {t}"
        assert torch.equal(act1[t, A:], act0[t, A:]), f"head direction, row {t}"


def test_row_zero_carries_the_real_head_direction(segment):
    """Zero SPEED, real HD[0].

    `actOffset` front-pads zeros and so cannot express this row; it is the
    reason the shift is built in `action_rows` instead of by the architecture.
    """
    obss, _, _ = segment
    _, act1, _, ad = _rows(1, segment)
    A = ad.num_acts
    assert act1[0, :A].sum() == 0, "row 0 must carry no action"
    assert act1[0, A + int(obss[0]["direction"])] == 1, "row 0 must carry HD[0]"
    assert act1[0, A:].sum() == 1, "exactly one head direction"


def test_the_segments_last_action_is_not_discarded(segment):
    """offset 1 gains a row, and it is a real one.

    The upstream `actOffset` pairs its front-pad with a tail-drop, which throws
    away a[L-1]. Building the rows here keeps it.
    """
    obss, acts, last = segment
    _, act0, _, ad = _rows(0, segment)
    _, act1, _, _ = _rows(1, segment)
    A = ad.num_acts
    assert len(act1) == len(act0) + 1
    tail = act1[-1]
    assert (tail[:A].sum() == 1) == (acts[-1] == FORWARD_IDX)
    assert tail[A + int(last["direction"])] == 1, "the tail row carries HD[L]"


@pytest.mark.parametrize("offset", (0, 1))
def test_boundary_bootstrap_reads_the_new_episode(offset):
    """At an episode cut the policy's first SR must come from the NEW episode.

    `collect_rollout` used to reset the tracker BEFORE the environment. Under
    offset 0 that is invisible - `init_sr` returns zeros and never reads the
    observation - but under offset 1 it silently built h[0] from the finished
    episode's last view, at a position the agent had already left.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = _env(seed=SEED + 1)
    pN = _net(env)
    obs_space, pre = get_obss_preprocessor(env.observation_space)
    ac = ACModelSR(obs_space, env.action_space, HIDDEN, False, True, True)
    algo = PredictivePPOAlgo(
        env, ac, pN, torch.device("cpu"), num_frames=2 * L, prnn_seqdur=L,
        pastSR=offset == 0, curious_agent=True, reward_alignment="next_obs",
        train_pN=False, epochs=1, batch_size=2 * L, preprocess_obss=pre, noise_std=0.0,
    )
    exps, _ = algo.collect_experiences()
    ad = algo.adapter
    first_of_episode_2 = {
        "image": exps.obs.image[L].to(torch.uint8).numpy(),
        "direction": int(exps.obs.direction[L]),
    }

    ad.reset_state()
    from_new = ad.init_sr(first_of_episode_2).squeeze(0)
    ad.reset_state()
    from_stale = ad.init_sr(algo.last_observations[0]).squeeze(0)

    assert not np.array_equal(
        np.asarray(first_of_episode_2["image"]),
        np.asarray(algo.last_observations[0]["image"]),
    ), "the two observations must differ or this test proves nothing"

    if offset == 0:
        assert (algo.SRs[L] == 0).all(), "offset 0 starts an episode from zeros"
    else:
        assert torch.allclose(algo.SRs[L], from_new, atol=1e-6)
        assert not torch.allclose(algo.SRs[L], from_stale, atol=1e-6)


@pytest.mark.parametrize("offset", (0, 1))
def test_batched_tracker_matches_two_serial_streams(offset):
    """B=2 batched == two B=1 serial trajectories, row 0 and boundaries included.

    Row 0 is the point: at offset 1 the batched shim has to build h[0] the same
    way `init_sr` does - a phase-0 masked step carrying (no action, HD[0]) - and
    it has to do it again at every episode reset. Zero noise, so a difference is
    the code and not a draw.
    """
    from curious_george.models.prnn_adapter import BatchedSRTrackerShim, SingleSRTracker

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    B, T, CUT = 2, 8, 4
    envs = [_env(seed=SEED + i) for i in range(B)]
    pN = _net(envs[0])
    ad = PRNNAdapter(pN, torch.device("cpu"), pastSR=offset == 0)

    rng = np.random.default_rng(SEED)
    obss = [[e.reset()] for e in envs]
    acts = [[] for _ in range(B)]
    for b, e in enumerate(envs):
        for _ in range(T):
            a = int(rng.choice(4))
            acts[b].append(a)
            obss[b].append(e.step(np.array([a]))[0])

    batched = BatchedSRTrackerShim(ad, [obss[b][0] for b in range(B)])
    batched_srs = [batched.initial_sr().clone()]
    for t in range(T):
        det = np.array([acts[b][t] for b in range(B)])
        pre = [obss[b][t] for b in range(B)]
        post = [obss[b][t + 1] for b in range(B)]
        batched_srs.append(batched.step(det, pre, post).clone())
        if t + 1 == CUT:  # a synchronized episode cut, as the collector makes
            rows = [batched.reset_env(b, obss[b][t + 1]) for b in range(B)]
            batched_srs[-1] = torch.cat(rows)

    for b in range(B):
        ad.reset_state()
        serial = SingleSRTracker(ad, obss[b][0])
        expected = [serial.initial_sr().clone()]
        for t in range(T):
            det = np.array([acts[b][t]])
            expected.append(serial.step(det, [obss[b][t]], [obss[b][t + 1]]).clone())
            if t + 1 == CUT:
                expected[-1] = serial.reset_env(0, obss[b][t + 1]).clone()
        for t, want in enumerate(expected):
            assert torch.allclose(batched_srs[t][b], want[0], atol=1e-6), (
                f"stream {b}, step {t}"
            )


@pytest.mark.parametrize("offset", (0, 1))
def test_device_backend_matches_the_cpu_table_at_either_offset(offset):
    """The fast path must compute the same rollout as the reference one.

    `test_device_collector.py` makes this comparison across `reward_alignment`
    but not across `action_offset`, and the A/B runs used the device backend -
    so the combination that actually trained was the one nothing checked. The
    offset touches `prediction_mses_device`, `train_on_episodes_batched`,
    `step_device` and the captured rollout body, none of which the serial tests
    above reach.
    """
    import dataclasses

    from curious_george.configs import EnvBackend
    from curious_george.training.setup import setup_training
    from tests.small_config import small_config
    from tests.test_device_collector import (
        _assert_rollouts_equal, _rng_state, _set_rng_state,
    )

    def build(device_env):
        cfg = small_config(
            backend=EnvBackend.DEVICE if device_env else EnvBackend.SERIAL_TABLE
        )
        return setup_training(dataclasses.replace(
            cfg, arch_prnn=dataclasses.replace(cfg.arch_prnn, action_offset=offset)
        )).algo

    reference, device = build(False), build(True)
    reference_rng, device_rng = _rng_state(reference.device), _rng_state(device.device)
    try:
        for _ in range(2):
            _set_rng_state(reference.device, reference_rng)
            expected, expected_logs = reference.collect_experiences()
            reference_rng = _rng_state(reference.device)

            _set_rng_state(device.device, device_rng)
            actual, actual_logs = device.collect_experiences()
            device_rng = _rng_state(device.device)

            _assert_rollouts_equal(
                reference, device, expected, actual, expected_logs, actual_logs
            )
    finally:
        device.envs.close()
        for shell in reference.envs:
            shell.env.close()
