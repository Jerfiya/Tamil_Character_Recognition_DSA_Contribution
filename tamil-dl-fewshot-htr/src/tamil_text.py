"""Tamil script helpers: Unicode normalisation, label parsing, and text rendering.

``cv2.putText`` and matplotlib's own text engine cannot shape Tamil (pre-base vowel
signs, split vowels, ligatures such as கு / கூ). Everything that must show a Tamil
label goes through Pillow (with the Raqm/HarfBuzz layout engine when available) and
is placed into matplotlib figures as a small image.
"""
from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

log = logging.getLogger(__name__)

# --- alphabet knowledge (used for reporting only; classes are always discovered) -------------
AYTHAM = "\u0b83"
VOWELS: List[str] = list("அஆஇஈஉஊஎஏஐஒஓஔ")
CONSONANTS: List[str] = list("கஙசஞடணதநபமயரலவழளறன")
PULLI = "\u0bcd"
# vowel signs in the order of VOWELS ('' = inherent a)
VOWEL_SIGNS: List[str] = ["", "\u0bbe", "\u0bbf", "\u0bc0", "\u0bc1", "\u0bc2", "\u0bc6", "\u0bc7",
                          "\u0bc8", "\u0bca", "\u0bcb", "\u0bcc"]
VARIANT_AU = "\u0bc6\u0bb3"  # ெ + ள (letter LLA) typed instead of the AU sign ௌ
VARIANT_KEY = "ெள*"


def nfc(text: str) -> str:
    """Return the NFC-normalised form of ``text``."""
    return unicodedata.normalize("NFC", text)


def parse_label(label: str) -> Dict[str, Optional[str]]:
    """Decompose a class label.

    Returns a dict with ``kind`` in {aytham, vowel, consonant, compound, variant, other},
    plus ``consonant`` and ``vowel`` (the vowel sign, '' for inherent a) when applicable.
    ``variant`` marks consonant + ெ + ள, a possible misspelling of consonant + ௌ; it is
    reported, never merged.
    """
    s = nfc(label)
    out: Dict[str, Optional[str]] = {"kind": "other", "consonant": None, "vowel": None}
    if s == AYTHAM:
        out["kind"] = "aytham"
    elif s in VOWELS:
        out["kind"] = "vowel"
        out["vowel"] = s
    elif len(s) >= 1 and s[0] in CONSONANTS:
        rest = s[1:]
        out["consonant"] = s[0]
        if rest == PULLI:
            out["kind"] = "consonant"
        elif rest in VOWEL_SIGNS:
            out["kind"] = "compound"
            out["vowel"] = rest
        elif rest == VARIANT_AU:
            out["kind"] = "variant"
            out["vowel"] = VARIANT_KEY
    return out


def english_category(label: str) -> str:
    """Coarse category name used in reports."""
    kind = parse_label(label)["kind"]
    return {"aytham": "aytham", "vowel": "vowel", "consonant": "consonant",
            "compound": "consonant-vowel", "variant": "consonant-vowel"}.get(kind or "", "other")


def expected_grid() -> List[str]:
    """All 18 x 12 consonant-vowel compounds in NFC."""
    return [nfc(c + v) for c in CONSONANTS for v in VOWEL_SIGNS]


def grid_report(labels: Iterable[str]) -> Dict[str, List[str]]:
    """Compare discovered labels with the standard alphabet."""
    present = {nfc(x) for x in labels}
    variants = sorted(x for x in present if parse_label(x)["kind"] == "variant")
    return {
        "missing_compounds": [x for x in expected_grid() if x not in present],
        "missing_vowels": [v for v in VOWELS if v not in present],
        "missing_consonants": [nfc(c + PULLI) for c in CONSONANTS if nfc(c + PULLI) not in present],
        "missing_aytham": [] if AYTHAM in present else [AYTHAM],
        "spelling_variants": variants,
        "variant_of": [nfc(x[0] + "\u0bcc") for x in variants],
        "unrecognised": sorted(x for x in present if parse_label(x)["kind"] == "other"),
    }


def vowel_sign_name(sign: Optional[str]) -> str:
    """Printable name for a vowel sign column (dotted circle carrier)."""
    if sign is None:
        return "-"
    if sign == "":
        return "அ"
    if sign == VARIANT_KEY:
        return VARIANT_KEY
    return VOWELS[VOWEL_SIGNS.index(sign)] if sign in VOWEL_SIGNS else sign


# --- fonts ----------------------------------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Nirmala.ttc", r"C:\Windows\Fonts\Nirmala.ttf", r"C:\Windows\Fonts\NirmalaUI.ttf",
    r"C:\Windows\Fonts\NotoSansTamil-Regular.ttf", r"C:\Windows\Fonts\latha.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
    "/System/Library/Fonts/Supplemental/Tamil MN.ttc",
]


@lru_cache(maxsize=1)
def find_tamil_font() -> Optional[str]:
    """Path of a font with Tamil glyphs (Nirmala UI on Windows, Noto Sans Tamil elsewhere)."""
    for cand in _FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    try:
        from matplotlib import font_manager
        for f in font_manager.findSystemFonts():
            name = Path(f).name.lower()
            if "tamil" in name or "nirmala" in name or "latha" in name:
                return f
    except Exception:  # pragma: no cover - matplotlib optional here
        pass
    log.warning("No Tamil font found; labels will not render correctly")
    return None


@lru_cache(maxsize=1)
def has_raqm() -> bool:
    """True when Pillow can do complex text layout."""
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


def _visual_order(text: str) -> str:
    """Poor-man's shaping used only without Raqm: move pre-base vowel signs before the consonant."""
    out: List[str] = []
    for ch in nfc(text):
        if ch in "\u0bc6\u0bc7\u0bc8" and out:
            out.insert(len(out) - 1, ch)
        elif ch == "\u0bca" and out:
            out.insert(len(out) - 1, "\u0bc6"); out.append("\u0bbe")
        elif ch == "\u0bcb" and out:
            out.insert(len(out) - 1, "\u0bc7"); out.append("\u0bbe")
        elif ch == "\u0bcc" and out:
            out.insert(len(out) - 1, "\u0bc6"); out.append("\u0bd7")
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=64)
def get_font(px: int) -> ImageFont.FreeTypeFont:
    """Load the Tamil font at ``px`` pixels."""
    path = find_tamil_font()
    engine = ImageFont.Layout.RAQM if has_raqm() else ImageFont.Layout.BASIC
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, px, layout_engine=engine)


@lru_cache(maxsize=1)
def _hb_fonts():
    """(HarfBuzz font, FreeType face) for the Tamil font, or None when the libraries are missing."""
    path = find_tamil_font()
    if path is None:
        return None
    try:
        import freetype
        import uharfbuzz as hb
        hb_face = hb.Face(hb.Blob.from_file_path(path), 0)
        return hb.Font(hb_face), freetype.Face(path, index=0), hb_face.upem
    except Exception as exc:
        log.warning("HarfBuzz/FreeType shaping unavailable (%s)", exc)
        return None


def shaping_engine() -> str:
    """Which engine renders Tamil: 'harfbuzz', 'raqm' or 'basic' (reordering only)."""
    if _hb_fonts() is not None:
        return "harfbuzz"
    return "raqm" if has_raqm() else "basic"


def _render_harfbuzz(text: str, px: int, color: Tuple[int, int, int], pad: int) -> Image.Image:
    """Shape with HarfBuzz, rasterise each glyph with FreeType, composite into an RGBA image."""
    import freetype
    import uharfbuzz as hb
    hb_font, ft_face, upem = _hb_fonts()
    buf = hb.Buffer()
    buf.add_str(nfc(text))
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)
    ft_face.set_pixel_sizes(0, px)
    scale = px / upem
    glyphs, pen = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        g = ft_face.glyph
        bmp = np.array(g.bitmap.buffer, dtype=np.uint8).reshape(g.bitmap.rows, g.bitmap.width) if g.bitmap.rows else None
        x = pen + pos.x_offset * scale + g.bitmap_left
        y = -(pos.y_offset * scale) - g.bitmap_top           # top edge relative to the baseline (down = +)
        if bmp is not None:
            glyphs.append((int(round(x)), int(round(y)), bmp))
        pen += pos.x_advance * scale
    if not glyphs:
        return Image.new("RGBA", (max(1, int(pen)) + 2 * pad, px), (255, 255, 255, 0))
    x0 = min(g[0] for g in glyphs); y0 = min(g[1] for g in glyphs)
    x1 = max(g[0] + g[2].shape[1] for g in glyphs); y1 = max(g[1] + g[2].shape[0] for g in glyphs)
    alpha = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), np.uint8)
    for x, y, bmp in glyphs:
        ys, xs = y - y0 + pad, x - x0 + pad
        region = alpha[ys:ys + bmp.shape[0], xs:xs + bmp.shape[1]]
        np.maximum(region, bmp, out=region)
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = color[0], color[1], color[2], alpha
    return Image.fromarray(rgba, "RGBA")


@lru_cache(maxsize=4096)
def render_text(text: str, px: int = 28, color: Tuple[int, int, int] = (0, 0, 0),
                pad: int = 2) -> Image.Image:
    """Render ``text`` to a tight RGBA image with correct Tamil shaping."""
    if _hb_fonts() is not None:
        return _render_harfbuzz(text, px, color, pad)
    font = get_font(px)
    s = nfc(text) if has_raqm() else _visual_order(text)
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    left, top, right, bottom = probe.textbbox((0, 0), s, font=font)
    w, h = max(1, right - left) + 2 * pad, max(1, bottom - top) + 2 * pad
    img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    ImageDraw.Draw(img).text((pad - left, pad - top), s, font=font, fill=color + (255,))
    return img


def draw_text_on_image(img: Image.Image, xy: Tuple[int, int], text: str, px: int = 28,
                       color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    """Paste shaped Tamil text onto a Pillow image in place (used by the webcam overlay)."""
    patch = render_text(text, px, color)
    img.paste(patch, (int(xy[0]), int(xy[1])), patch)


# --- matplotlib integration -----------------------------------------------------------------
def setup_matplotlib() -> None:
    """Register the Tamil font with matplotlib (glyph fallback) and set sane defaults."""
    import matplotlib
    from matplotlib import font_manager
    path = find_tamil_font()
    families = ["DejaVu Sans"]
    if path:
        try:
            font_manager.fontManager.addfont(path)
            families.append(font_manager.FontProperties(fname=path).get_name())
        except Exception as exc:  # pragma: no cover
            log.warning("could not register %s with matplotlib: %s", path, exc)
    matplotlib.rcParams["font.family"] = families
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["savefig.dpi"] = 160
    matplotlib.rcParams["figure.dpi"] = 100


def mpl_text(ax, x: float, y: float, text: str, px: int = 22, color=(0, 0, 0), zoom: float = 0.5,
             xycoords="data", box_alignment=(0.5, 0.5), **kw):
    """Place shaped Tamil ``text`` at (x, y) on a matplotlib axis as an image annotation."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    arr = np.asarray(render_text(text, px, tuple(int(c) for c in color)))
    ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), (x, y), xycoords=xycoords, frameon=False,
                        box_alignment=box_alignment, pad=0, annotation_clip=False, **kw)
    ax.add_artist(ab)
    return ab


def tamil_ticks(ax, labels: Sequence[str], axis: str = "x", positions: Optional[Sequence[float]] = None,
                px: int = 22, zoom: float = 0.5, offset: float = 0.012, color=(0, 0, 0)) -> None:
    """Replace tick labels on ``axis`` with shaped Tamil label images."""
    pos = list(positions) if positions is not None else list(range(len(labels)))
    if axis == "x":
        ax.set_xticks(pos); ax.set_xticklabels([""] * len(pos))
        tr = ax.get_xaxis_transform()
        for p, lab in zip(pos, labels):
            mpl_text(ax, p, -offset, lab, px, color, zoom, xycoords=tr, box_alignment=(0.5, 1.0))
    else:
        ax.set_yticks(pos); ax.set_yticklabels([""] * len(pos))
        tr = ax.get_yaxis_transform()
        for p, lab in zip(pos, labels):
            mpl_text(ax, -offset, p, lab, px, color, zoom, xycoords=tr, box_alignment=(1.0, 0.5))
