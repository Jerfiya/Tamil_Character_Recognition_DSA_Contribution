"""Preprocessing: output contract, polarity invariance, train/inference parity."""
import numpy as np
import pandas as pd

from conftest import ROOT, needs_data
from preprocessing import preprocess_gray, preprocess_path, to_tensor


def _glyph(dark_on_light=True, size=(90, 70)):
    g = np.full(size, 230, np.uint8)
    g[20:70, 30:36] = 20
    g[40:46, 15:60] = 20
    return g if dark_on_light else 255 - g


def test_shape_range(cfg):
    out = preprocess_gray(_glyph(), cfg.preprocess, 48)
    assert out.shape == (48, 48) and out.dtype == np.uint8
    t = to_tensor(out, cfg.preprocess)
    assert t.shape == (1, 48, 48) and t.min() >= -1.0 - 1e-6 and t.max() <= 1.0 + 1e-6
    assert preprocess_gray(_glyph(), cfg.preprocess, 96).shape == (96, 96)


def test_ink_white_on_black_and_polarity_invariant(cfg):
    a = preprocess_gray(_glyph(True), cfg.preprocess, 48)
    b = preprocess_gray(_glyph(False), cfg.preprocess, 48)
    assert a[0, 0] == 0 and a.max() > 200           # background black, ink bright
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 10


def test_blank_image_gives_zeros(cfg):
    assert preprocess_gray(np.full((50, 50), 200, np.uint8), cfg.preprocess, 48).max() == 0


def test_aspect_preserved(cfg):
    g = np.full((100, 100), 255, np.uint8)
    g[10:90, 48:52] = 0                              # tall thin stroke stays tall and thin
    out = preprocess_gray(g, cfg.preprocess, 48)
    ys, xs = np.where(out > 0)
    assert (ys.max() - ys.min()) > 3 * (xs.max() - xs.min())


@needs_data
def test_train_inference_parity(cfg):
    """The cached training tensor equals preprocessing the original file at inference time."""
    man = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8-sig", keep_default_na=False)
    imgs = np.load(ROOT / "data" / "processed" / "images_48.npy", mmap_mode="r")
    for i in np.random.default_rng(0).choice(len(man), 20, replace=False):
        live = preprocess_path(man.path.iat[i], cfg.preprocess, 48)
        assert np.array_equal(live, imgs[int(man.idx.iat[i])]), man.path.iat[i]
