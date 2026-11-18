"""Backbones and methods: output shapes on dummy batches, prototype math, losses."""
import math

import pytest
import torch

from backbones import build_backbone
from config import load_config
from fewshot import EpisodeInfo, build_method
from fewshot.transductive import transductive_logits
from losses import batch_hard_triplet, contrastive_loss, margin_logits, proto_logits, prototypes, sq_euclidean

BACKBONES = ["conv4", "resnet12", "resnet18", "vit_small", "efficientnet_b0", "mobilenetv3_large"]
METHODS = ["protonet", "matchingnet", "relationnet", "siamese", "arcface", "baselinepp", "maml", "supervised"]


def _cfg(method="protonet", backbone="conv4", **kw):
    over = [f"method.name={method}", f"model.backbone={backbone}", "model.pretrained=false", "threads=2"]
    over += [f"{k}={v}" for k, v in kw.items()]
    return load_config(None, over)


@pytest.mark.parametrize("name", BACKBONES)
def test_backbone_shapes(name):
    cfg = _cfg(backbone=name)
    bb = build_backbone(cfg.model).eval()
    x = torch.randn(3, 1, 48, 48)
    with torch.no_grad():
        f = bb.features(x)
        e = bb(x)
    assert f.ndim == 4 and f.shape[:2] == (3, bb.feat_channels)
    assert e.shape == (3, cfg.model.emb_dim) and torch.isfinite(e).all()


@pytest.mark.parametrize("method", METHODS)
def test_method_episode_shapes_and_loss(method):
    n, k, q = 4, 2, 3
    model = build_method(_cfg(method), n_train_classes=10)
    x = torch.randn(n * k + n * q, 1, 48, 48)
    qy = torch.arange(n).repeat_interleave(q)
    local = torch.cat([torch.arange(n).repeat_interleave(k), qy])
    info = EpisodeInfo(n, k, qy, local, local + 3, 0.05)
    model.train()
    out = model.train_loss_images(x, info) if model.needs_images else model.train_loss(model.features(x), info)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    model.eval()
    with torch.no_grad() if not model.needs_images else torch.enable_grad():
        if model.needs_images:
            lg = model.episode_logits_images(x[: n * k], x[n * k:], n, k)
        else:
            f = model.features(x)
            lg = model.episode_logits(f[: n * k], f[n * k:], n, k)
    assert lg.shape == (n * q, n)


def test_prototype_math():
    s = torch.tensor([[0., 0.], [2., 0.], [10., 10.], [12., 10.]])       # 2 classes x 2 shots
    p = prototypes(s, 2, 2)
    assert torch.allclose(p, torch.tensor([[1., 0.], [11., 10.]]))
    q = torch.tensor([[1., 1.], [11., 9.]])
    lg = proto_logits(s, q, 2, 2)
    assert torch.allclose(lg[0, 0], torch.tensor(-1.0)) and lg.argmax(1).tolist() == [0, 1]
    a, b = torch.randn(5, 7), torch.randn(4, 7)
    assert torch.allclose(sq_euclidean(a, b), torch.cdist(a, b) ** 2, atol=1e-4)


def test_protonet_temperature_scaling():
    m = build_method(_cfg("protonet"))
    s, q = torch.randn(6, 128), torch.randn(4, 128)
    with torch.no_grad():
        m.log_temp.fill_(math.log(2.0))
        a = m.episode_logits(s, q, 3, 2)
        m.log_temp.fill_(0.0)
        b = m.episode_logits(s, q, 3, 2)
    assert torch.allclose(a, 2 * b, atol=1e-5)


def test_metric_losses():
    emb = torch.nn.functional.normalize(torch.randn(12, 8), dim=1).requires_grad_()
    y = torch.arange(4).repeat_interleave(3)
    for fn in (batch_hard_triplet, contrastive_loss):
        loss, comps = fn(emb, y, 0.3)
        assert torch.isfinite(loss) and {"dist_pos", "dist_neg", "loss_pos", "loss_neg"} <= set(comps)
    # perfectly separated embeddings give zero triplet loss
    sep = torch.eye(4).repeat_interleave(3, 0) * 5
    assert float(batch_hard_triplet(sep, y, 0.3)[0]) == 0.0
    cos = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    ml = margin_logits(cos, torch.tensor([0, 1]), 10.0, 0.3)
    assert ml[0, 0] < 9.0 and ml[0, 1] == pytest.approx(1.0)


def test_transductive_shapes():
    s, q = torch.randn(10, 16), torch.randn(15, 16)
    lg = transductive_logits(s, q, 5, 2, 3, "euclidean", 1.0)
    assert lg.shape == (15, 5)
