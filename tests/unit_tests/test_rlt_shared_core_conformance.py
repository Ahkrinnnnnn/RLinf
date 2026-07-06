# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-framework conformance test: RLinf reuses lerobot's shared RLT core.

This test proves the **dual-framework cooperation** (Path 1):

1. RLinf can import lerobot's framework-agnostic RLT core
   (``lerobot.rlt.shared``) — networks + loss math + replay logic.
2. Running the shared loss math inside RLinf's environment reproduces lerobot's
   :func:`golden_loss_values` **bit-for-bit** on CPU (same code path, same
   deterministic fixture).
3. RLinf's :class:`RLTTD3Driver` drives the shared math end-to-end (a TD3 outer
   step with UTD, delayed actor update, polyak target EMA) and produces finite,
   well-shaped stats.
4. Actor weights round-trip through ``torch.save``/``load`` — the mechanism for
   sim-to-real transfer (RLinf-trained actor loads into a lerobot actor).
5. The chunked rollout runner produces lerobot-compatible ``ReplayWindow``
   transitions.

Run with RLinf's py3.12 ``.venv`` (which has the lerobot fork installed
editable)::
    .venv/bin/pytest tests/unit_tests/test_rlt_shared_core_conformance.py -q
"""

import numpy as np  # noqa: F401  (asserted importable in RLinf env)
import pytest
import torch

from lerobot.rlt.shared import (  # noqa: E402
    ChunkActor,
    ChunkCriticEnsemble,
    golden_loss_values,
    make_rlt_tiny_fixture,
    polyak_update,
    rlt_actor_loss,
    td3_critic_loss,
)
from lerobot.rlt.replay_windows import ReplayWindow  # noqa: E402

from rlinf.rlt import (  # noqa: E402
    ChunkedRolloutRunner,
    MockChunkEnv,
    RLTTD3Driver,
    RLTTD3DriverConfig,
)
from rlinf.rlt.chunked_rollout import make_mock_actor_chunk, make_mock_machine_a, run_mock_episode  # noqa: E402


# ---------------------------------------------------------------------------
# (1) Importability of the shared core inside the RLinf environment
# ---------------------------------------------------------------------------


def test_shared_core_importable_in_rlinf_env():
    fx = make_rlt_tiny_fixture()
    assert isinstance(fx["actor"], ChunkActor)
    assert isinstance(fx["critic_ensemble"], ChunkCriticEnsemble)
    assert isinstance(fx["critic_target"], ChunkCriticEnsemble)


# ---------------------------------------------------------------------------
# (2) Bit-for-bit identity with lerobot's golden loss values
# ---------------------------------------------------------------------------


def test_shared_math_matches_golden_bit_for_bit():
    fx = make_rlt_tiny_fixture()
    meta = fx["meta"]

    loss_critic = td3_critic_loss(
        fx["critic_ensemble"], fx["critic_target"], fx["actor"], fx["fb"], meta["discount"]
    )
    loss_actor, info = rlt_actor_loss(
        fx["actor"],
        fx["critic_ensemble"],
        fx["fb"],
        chunk_len=meta["chunk_len"],
        action_dim=meta["action_dim"],
        bc_weight=meta["bc_weight"],
        delta_penalty_weight=meta["delta_penalty_weight"],
        ref_dropout_prob=meta["ref_dropout_prob"],
        in_warmup=meta["in_warmup"],
    )

    golden = golden_loss_values()
    # CPU, deterministic fixture, same code path → bit-identical floats.
    assert float(loss_critic.detach().item()) == pytest.approx(golden["loss_critic"], abs=0.0, rel=0.0)
    assert float(loss_actor.detach().item()) == pytest.approx(golden["loss_actor"], abs=0.0, rel=0.0)
    assert info["rl_loss"] == pytest.approx(golden["rl_loss"], abs=0.0, rel=0.0)
    assert info["bc_loss"] == pytest.approx(golden["bc_loss"], abs=0.0, rel=0.0)
    assert info["delta_penalty"] == pytest.approx(golden["delta_penalty"], abs=0.0, rel=0.0)


def test_golden_loss_values_are_stable_in_rlinf_env():
    a = golden_loss_values()
    b = golden_loss_values()
    assert a == b


# ---------------------------------------------------------------------------
# (3) RLinf's TD3 driver drives the shared math end-to-end
# ---------------------------------------------------------------------------


def _make_driver(device: str = "cpu") -> tuple[RLTTD3Driver, dict]:
    from lerobot.policies.rlt_actor import RLTActorConfig
    from lerobot.policies.rlt_actor.configuration_rlt_actor import ChunkActorNetworkConfig

    cfg = RLTActorConfig(
        chunk_len=3,
        action_dim=2,
        token_dim=8,
        proprio_dim=4,
        fixed_std=0.1,
        ref_dropout_prob=0.0,
        actor_network_kwargs=ChunkActorNetworkConfig(
            hidden_dims=[4, 4], activate_final=True, activations="SiLU"
        ),
    )
    driver_cfg = RLTTD3DriverConfig(
        critic_hidden_dims=(4, 4),
        utd_ratio=2,
        policy_update_freq=2,
        bc_weight_max=0.7,
        bc_decay_steps=1000,
        delta_penalty_weight=0.05,
    )
    driver = RLTTD3Driver(cfg, driver_cfg, device=device)
    return driver, _make_fb(cfg)


def _make_fb(cfg) -> dict:
    b = 4
    cd = cfg.chunk_len * cfg.action_dim
    from lerobot.rlt.replay_windows import SOURCE_BASE, SOURCE_HUMAN, SOURCE_MIXED, SOURCE_RL
    from lerobot.utils.constants import ACTION

    return {
        ACTION: torch.randn(b, cd),
        "reward": torch.randn(b),
        "done": torch.zeros(b),
        "state": {"z_rl": torch.randn(b, cfg.token_dim), "proprio": torch.randn(b, cfg.proprio_dim)},
        "next_state": {"z_rl": torch.randn(b, cfg.token_dim), "proprio": torch.randn(b, cfg.proprio_dim)},
        "ref_chunk": torch.randn(b, cd),
        "ref_chunk_next": torch.randn(b, cd),
        "source": torch.tensor([SOURCE_BASE, SOURCE_RL, SOURCE_HUMAN, SOURCE_MIXED]),
        "bc_target": torch.randn(b, cd),
    }


def test_driver_update_step_produces_finite_stats_and_advances():
    driver, fb = _make_driver()
    s0 = driver.update_step(fb)
    assert "loss_critic" in s0 and np.isfinite(s0["loss_critic"])
    # step 0, policy_update_freq=2 → actor updated.
    assert "loss_actor" in s0 and np.isfinite(s0["loss_actor"])
    assert "rl_loss" in s0 and "bc_loss" in s0 and "delta_penalty" in s0
    assert driver._step == 1

    s1 = driver.update_step(fb)
    # step 1 → actor NOT updated.
    assert "loss_actor" not in s1
    assert driver._step == 2


def test_driver_critic_weights_change_after_update():
    driver, fb = _make_driver()
    before = {n: p.clone() for n, p in driver.critic_ensemble.named_parameters()}
    driver.update_step(fb)
    assert any(not torch.equal(p, before[n]) for n, p in driver.critic_ensemble.named_parameters())


def test_driver_target_tracks_online_via_polyak():
    driver, fb = _make_driver()
    online_before = {n: p.clone() for n, p in driver.critic_ensemble.named_parameters()}
    target_before = {n: p.clone() for n, p in driver.critic_target.named_parameters()}
    driver.update_step(fb)
    moved_online = any(
        not torch.equal(p, online_before[n]) for n, p in driver.critic_ensemble.named_parameters()
    )
    moved_target = any(
        not torch.equal(p, target_before[n]) for n, p in driver.critic_target.named_parameters()
    )
    assert moved_online and moved_target


# ---------------------------------------------------------------------------
# (4) Actor weight round-trip (sim ↔ real transfer mechanism)
# ---------------------------------------------------------------------------


def test_actor_weight_round_trip_via_torch_save(tmp_path):
    driver_a, _ = _make_driver()
    fb = _make_fb(driver_a.actor_config)
    # Train a couple of steps so weights diverge from init.
    driver_a.update_step(fb)
    driver_a.update_step(fb)

    w = driver_a.get_actor_weights()
    save_path = tmp_path / "actor.pt"
    torch.save(w, save_path)

    driver_b, _ = _make_driver()
    loaded = torch.load(save_path, weights_only=True)
    driver_b.load_actor_weights(loaded)

    z = torch.randn(2, driver_a.actor_config.token_dim)
    p = torch.randn(2, driver_a.actor_config.proprio_dim)
    r = torch.randn(2, driver_a.actor_config.chunk_len * driver_a.actor_config.action_dim)
    with torch.no_grad():
        out_a = driver_a.actor(z, p, r)[2]
        out_b = driver_b.actor(z, p, r)[2]
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# (5) Chunked rollout → lerobot-compatible replay windows
# ---------------------------------------------------------------------------


def test_mock_episode_produces_well_formed_windows():
    ws = run_mock_episode(chunk_len=3, action_dim=2, token_dim=8, proprio_dim=4, max_chunks=4, stride=0)
    assert len(ws) >= 1
    for w in ws:
        assert isinstance(w, ReplayWindow)
        assert w.action.shape == (3 * 2,)
        assert w.state["z_rl"].shape == (8,)
        assert w.state["proprio"].shape == (4,)
        assert "ref_chunk" in w.complementary_info
        assert "ref_chunk_next" in w.complementary_info
        assert "source" in w.complementary_info
        assert "bc_target" in w.complementary_info


def test_chunked_rollout_runner_with_actor_chunk_fn():
    env = MockChunkEnv(proprio_dim=4, action_dim=2, max_steps=12)
    runner = ChunkedRolloutRunner(
        chunk_len=3, action_dim=2, token_dim=8, proprio_dim=4, stride=0
    )
    machine_a = make_mock_machine_a(8, 3, 2)
    actor = make_mock_actor_chunk()  # returns ref chunk → SOURCE_BASE-like content
    ws = runner.run_episode_with_env(machine_a, actor, env, max_chunks=4)
    assert len(ws) >= 1


def test_polyak_update_shared_symbol_callable_from_rlinf():
    src = torch.nn.Linear(2, 2)
    tgt = torch.nn.Linear(2, 2)
    with torch.no_grad():
        src.weight.fill_(1.0)
        tgt.weight.fill_(0.0)
    polyak_update(tgt, src, tau=0.5)
    assert torch.allclose(tgt.weight, torch.full_like(tgt.weight, 0.5))
