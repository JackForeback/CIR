"""Tests for the alternating-decoder VAE family."""

import os

import pytest
import torch

from cir.cli import load_config
from cir.data.mnist import default_data_root
from cir.experiments.alternating import AlternatingVAEExperiment
from cir.experiments.registry import get_experiment
from cir.models.alternating import (
    VARIANTS,
    AddedLossVAE,
    AlternatingVAE,
    FOLVAE,
    LAVAE,
    build_linear_path,
)
from cir.train import run_from_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

SHAPE = dict(input_dim=24, latent_dim=4, encoder_layers=[8], decoder_layers=[8], linear_layers=[6])


# --- the linear path ---------------------------------------------------------

def test_linear_path_is_affine():
    """No activations anywhere, which is the whole point of the second path."""
    path = build_linear_path(3, [5, 4], 6)
    a, b = torch.randn(1, 3), torch.randn(1, 3)
    origin = path(torch.zeros(1, 3))

    combined = path(a + b) - origin
    separate = (path(a) - origin) + (path(b) - origin)
    assert torch.allclose(combined, separate, atol=1e-5)


def test_linear_path_without_hidden_layers_is_a_single_map():
    path = build_linear_path(3, [], 6)
    assert len(path) == 1
    assert path(torch.randn(2, 3)).shape == (2, 6)


# --- alternation -------------------------------------------------------------

def test_schedule_picks_the_linear_path_on_every_other_step():
    model = LAVAE(**SHAPE)
    assert [model.uses_linear_path(s) for s in range(4)] == [True, False, True, False]


def test_schedule_period_is_configurable():
    model = LAVAE(**SHAPE, alternate_every=3)
    assert [model.uses_linear_path(s) for s in range(6)] == [True, False, False, True, False, False]


def test_alternate_every_must_be_positive():
    with pytest.raises(ValueError, match="alternate_every"):
        LAVAE(**SHAPE, alternate_every=0)


def test_decode_routes_through_the_path_the_step_selects():
    torch.manual_seed(0)
    model = LAVAE(**SHAPE).eval()
    z = torch.randn(2, SHAPE["latent_dim"])

    linear_out, linear_aux = model.decode(z, step=0)
    nonlinear_out, nonlinear_aux = model.decode(z, step=1)

    assert torch.allclose(linear_out, model.linear_decode(z))
    assert torch.allclose(nonlinear_out, model.decoder(z))
    # Plain alternation carries no auxiliary term.
    assert linear_aux.item() == 0.0 and nonlinear_aux.item() == 0.0


def test_forward_reports_which_path_it_used():
    model = LAVAE(**SHAPE)
    x = torch.rand(3, SHAPE["input_dim"])

    assert model(x, step=0)["used_linear_path"] is True
    out = model(x, step=1)
    assert out["used_linear_path"] is False
    assert out["x_hat"].shape == (3, SHAPE["input_dim"])
    assert set(out) >= {"x_hat", "kl_loss", "mu", "log_var", "z", "aux_loss"}


def test_the_linear_path_is_unbounded_unlike_the_sigmoid_decoder():
    """Squashing the linear path would make it nonlinear, so it is left raw."""
    model = LAVAE(**SHAPE)
    with torch.no_grad():
        model.linear_decoder[-1].bias.fill_(5.0)
    assert model.linear_decode(torch.zeros(1, SHAPE["latent_dim"])).max() > 1.0


# --- variants ----------------------------------------------------------------

def test_folvae_freezes_only_the_linear_path_output_layer():
    model = FOLVAE(**SHAPE)
    frozen = list(model.linear_decoder[-1].parameters())
    assert all(not p.requires_grad for p in frozen)
    assert all(p.requires_grad for p in model.linear_decoder[0].parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())

    model.decode(torch.randn(2, SHAPE["latent_dim"]), step=0)[0].sum().backward()
    assert all(p.grad is None for p in frozen)


def test_lavae_trains_both_paths():
    model = LAVAE(**SHAPE)
    assert all(p.requires_grad for p in model.linear_decoder.parameters())


def test_added_loss_variant_measures_decoder_disagreement():
    torch.manual_seed(0)
    model = AddedLossVAE(**SHAPE).eval()
    z = torch.randn(4, SHAPE["latent_dim"])

    x_hat, aux = model.decode(z, step=0)
    # Never switches paths, whatever the step.
    assert torch.allclose(x_hat, model.decoder(z))
    assert not model.uses_linear_path(0)

    expected = (model.decoder(z) - model.linear_decode(z)).pow(2).mean()
    assert torch.isclose(aux, expected, atol=1e-6)
    assert aux.item() >= 0
    aux.backward()  # must stay in the autograd graph


def test_every_variant_is_an_alternating_vae():
    assert set(VARIANTS) == {"lavae", "folvae", "added_loss"}
    for cls in VARIANTS.values():
        assert issubclass(cls, AlternatingVAE)


# --- the experiment ----------------------------------------------------------

def _cfg(tmp_path, **overrides):
    cfg = load_config(os.path.join(CONFIG_DIR, "altvae.yaml"))
    cfg.update(
        input_dim=784,
        epochs=1,
        train_subset=256,
        test_subset=128,
        device="cpu",
        output_dir=str(tmp_path),
        log_dir=str(tmp_path),
        log_every=0,
    )
    cfg.update(overrides)
    return cfg


def test_altvae_is_registered():
    assert get_experiment("altvae") is AlternatingVAEExperiment


def test_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="variant must be one of"):
        run_from_config(_cfg(tmp_path, variant="bogus"))


def test_the_optimizer_skips_a_frozen_output_layer(tmp_path):
    experiment = AlternatingVAEExperiment(_cfg(tmp_path, variant="folvae"))
    experiment.setup()
    optimized = sum(p.numel() for group in experiment.optimizer.param_groups for p in group["params"])
    total = sum(p.numel() for p in experiment.model.parameters())
    frozen = sum(p.numel() for p in experiment.model.linear_decoder[-1].parameters())
    assert optimized == total - frozen


def test_validation_never_walks_the_schedule(tmp_path):
    """Evaluation must decode nonlinearly so epochs stay comparable."""
    experiment = AlternatingVAEExperiment(_cfg(tmp_path))
    experiment.setup()
    batch = (torch.rand(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))

    experiment.model.eval()
    experiment.compute_loss(batch)
    assert experiment.global_step == 0

    experiment.model.train()
    experiment.compute_loss(batch)
    assert experiment.global_step == 1


@pytest.mark.skipif(
    not os.path.exists(os.path.join(default_data_root(), "MNIST")),
    reason="MNIST is not present under data/",
)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_variant_runs_an_epoch(tmp_path, variant):
    history = run_from_config(_cfg(tmp_path, variant=variant))
    assert len(history["train_loss"]) == 1
    assert torch.isfinite(torch.tensor(history["train_loss"][0]))
    assert torch.isfinite(torch.tensor(history["val_loss"][0]))
