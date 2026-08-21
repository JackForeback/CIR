"""End-to-end tests: config handling, the registry, models, and tiny runs."""

import glob
import json
import os

import pytest
import torch

from cir.cli import apply_overrides, load_config, set_nested
from cir.experiments.base import LOSS_FUNCTIONS, OPTIMIZERS, resolve_device
from cir.experiments.registry import EXPERIMENTS, get_experiment
from cir.models.alvae import ALVAE
from cir.models.vae import VAE, build_mlp, get_activation
from cir.train import run_from_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


# --- config ------------------------------------------------------------------

def test_overrides_parse_yaml_types_including_nested_keys():
    cfg = {"epochs": 1, "flags": {"evo_weights": False}}
    apply_overrides(cfg, ["epochs=5", "lr=1e-3", "flags.evo_weights=true", "scalars=[1,2,3]"])
    assert cfg["epochs"] == 5 and isinstance(cfg["epochs"], int)
    assert cfg["lr"] == pytest.approx(1e-3)
    assert cfg["flags"]["evo_weights"] is True
    assert cfg["scalars"] == [1, 2, 3]


def test_set_nested_creates_missing_levels():
    cfg = {}
    set_nested(cfg, "a.b.c", 1)
    assert cfg == {"a": {"b": {"c": 1}}}


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides({}, ["epochs"])


# Derived from the registry rather than hard-coded, so registering an
# experiment without shipping a config for it fails here.
@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_every_registered_experiment_ships_a_config(name):
    cfg = load_config(os.path.join(CONFIG_DIR, f"{name}.yaml"))
    assert cfg["experiment"] == name
    assert get_experiment(cfg["experiment"]) is EXPERIMENTS[name]


def test_configs_contain_no_absolute_paths():
    configs = glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
    assert configs
    for path in configs:
        raw = open(path).read()
        assert "/mnt/" not in raw and "/home/" not in raw, path


def test_unknown_experiment_reports_what_is_registered():
    with pytest.raises(KeyError, match="registered"):
        get_experiment("nope")


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_optimizer_and_loss_tables_are_constructible():
    model = torch.nn.Linear(2, 2)
    for factory in OPTIMIZERS.values():
        assert factory(model.parameters(), lr=0.01) is not None
    for factory in LOSS_FUNCTIONS.values():
        assert factory() is not None


# --- models ------------------------------------------------------------------

def test_vae_forward_shapes_and_a_finite_kl():
    model = VAE(input_dim=20, latent_dim=4, encoder_layers=[8], decoder_layers=[8])
    out = model(torch.rand(6, 20))
    assert out["x_hat"].shape == (6, 20)
    assert out["mu"].shape == out["log_var"].shape == out["z"].shape == (6, 4)
    assert torch.isfinite(out["kl_loss"]) and out["kl_loss"] >= 0
    # The decoder ends in a sigmoid, so reconstructions stay in the target range.
    assert (out["x_hat"] >= 0).all() and (out["x_hat"] <= 1).all()


def test_kl_is_zero_for_a_standard_normal_posterior():
    model = VAE(input_dim=4, latent_dim=3, encoder_layers=[], decoder_layers=[])
    kl = model.get_kl_loss(torch.zeros(5, 3), torch.zeros(5, 3))
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_reductions_relate_as_expected_and_reject_junk():
    model = VAE(input_dim=4, latent_dim=3, encoder_layers=[], decoder_layers=[])
    mu, log_var = torch.randn(8, 3), torch.randn(8, 3)
    total = model.get_kl_loss(mu, log_var, "sum")
    assert torch.isclose(model.get_kl_loss(mu, log_var, "batchmean"), total / 8, atol=1e-5)
    assert model.get_kl_loss(mu, log_var, "none").shape == (8, 3)
    with pytest.raises(ValueError, match="reduction"):
        model.get_kl_loss(mu, log_var, "average")


def test_build_mlp_and_activation_lookup():
    stack, width = build_mlp(10, [8, 4], "relu")
    assert width == 4 and stack(torch.randn(2, 10)).shape == (2, 4)
    assert build_mlp(10, [], "relu")[1] == 10  # empty stack is a pass-through
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("swish")


def test_alvae_adds_a_finite_auxiliary_loss():
    model = ALVAE(input_dim=32, latent_dim=4, encoder_layers=[8], decoder_layers=[8], num_basis=8)
    out = model(torch.rand(6, 32))
    assert "aux_loss" in out
    assert torch.isfinite(out["aux_loss"]) and out["aux_loss"] >= 0
    out["aux_loss"].backward()  # must stay in the autograd graph


def test_a_complete_basis_makes_the_auxiliary_loss_vanish():
    # num_basis == input_dim spans the whole space, so nothing is left over.
    model = ALVAE(input_dim=16, latent_dim=4, encoder_layers=[8], decoder_layers=[8], num_basis=16)
    assert torch.isclose(model.auxiliary_loss(torch.rand(4, 16)), torch.tensor(0.0), atol=1e-6)


# --- runs --------------------------------------------------------------------

def _linear_cfg(tmp_path, **overrides):
    cfg = load_config(os.path.join(CONFIG_DIR, "linear.yaml"))
    cfg.update(
        samples_per_class=120,
        num_training_steps=4,
        num_seeds=2,
        device="cpu",
        output_dir=str(tmp_path),
        log_dir=str(tmp_path),
        log_every=0,
    )
    cfg.update(overrides)
    return cfg


def test_linear_experiment_runs_and_writes_its_figures(tmp_path):
    result = run_from_config(_linear_cfg(tmp_path))

    assert set(result["mean_gap"]) == {"train", "test"}
    assert len(result["mean_gap"]["train"]) == 4
    assert all(0.0 <= g <= 1.0 for g in result["mean_gap"]["train"])

    for figure in ("sample_plot.png", "avg_accuracy.png", "avg_gap.png"):
        assert (tmp_path / figure).exists()
    assert json.loads((tmp_path / "results.json").read_text())


@pytest.mark.parametrize(
    "overrides",
    [
        {"apply_projection": True, "projection_mode": "shift", "target": "median"},
        {"apply_projection": True, "projection_mode": "scale", "target": "max"},
        {"flags": {"fairness_loss": "per_class_gap"}},
        {"flags": {"fairness_loss": "soft_accuracy_gap"}},
        {"flags": {"evo_weights": True}, "evo": {"pop_size": 40, "tournament_size": 20}},
        {"flags": {"plot_boundaries": True}},
    ],
    ids=["shift", "scale", "per_class_gap", "soft_gap", "evo", "boundaries"],
)
def test_every_linear_intervention_runs(tmp_path, overrides):
    result = run_from_config(_linear_cfg(tmp_path, **overrides))
    assert all(0.0 <= g <= 1.0 for g in result["mean_gap"]["test"])


def test_projection_toward_the_etf_narrows_the_class_gap(tmp_path):
    """The project's central claim, on a deliberately lopsided configuration."""
    common = dict(scalars=[1, 3, 1], num_training_steps=25, num_seeds=3, lr=0.005)
    baseline = run_from_config(_linear_cfg(tmp_path / "base", **common))
    projected = run_from_config(
        _linear_cfg(
            tmp_path / "proj",
            apply_projection=True,
            projection_mode="shift",
            target="median",
            **common,
        )
    )
    assert projected["final_gap"]["train"] <= baseline["final_gap"]["train"]


def test_linear_experiment_validates_per_class_list_lengths(tmp_path):
    with pytest.raises(ValueError, match="expected 3 per-class values"):
        run_from_config(_linear_cfg(tmp_path, scalars=[1, 2]))


def test_linear_experiment_rejects_an_unknown_fairness_loss(tmp_path):
    with pytest.raises(ValueError, match="flags.fairness_loss"):
        run_from_config(_linear_cfg(tmp_path, flags={"fairness_loss": "bogus"}))


@pytest.mark.parametrize("name", ["vae", "alvae"])
def test_vae_experiments_run_an_epoch_on_a_data_subset(tmp_path, name):
    cfg = load_config(os.path.join(CONFIG_DIR, f"{name}.yaml"))
    cfg.update(
        epochs=1,
        train_subset=256,
        test_subset=128,
        device="cpu",
        output_dir=str(tmp_path),
        log_dir=str(tmp_path),
        log_every=0,
    )
    if not os.path.exists(os.path.join(cfg.get("data_root") or _data_root(), "MNIST")):
        pytest.skip("MNIST is not present under data/")

    history = run_from_config(cfg)
    assert len(history["train_loss"]) == 1
    assert torch.isfinite(torch.tensor(history["train_loss"][0]))
    assert torch.isfinite(torch.tensor(history["val_loss"][0]))


def _data_root():
    from cir.data.mnist import default_data_root

    return default_data_root()
