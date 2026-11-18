"""Class discovery: nesting, empty folders, zips, unicode paths, NFC, source groups."""
import unicodedata
import zipfile

import cv2
import numpy as np

import tamil_text as tt
from dataset_audit import discover_classes, extract_zips, locate_dataset_root, source_group
from preprocessing import read_image_gray

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def _img(path, val=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.full((20, 20), 255, np.uint8)
    a[5:15, 8:12] = val
    ok, buf = cv2.imencode(path.suffix, a)
    buf.tofile(str(path))


def _tree(tmp_path):
    root = tmp_path / "scan" / "Tamil Letters (copy)"
    _img(root / "உயிர் எழுத்துக்கள" / "அ" / "a 1.png")
    _img(root / "உயிர் எழுத்துக்கள" / "அ" / "a (2).jpg")
    _img(root / "மெய் எழுத்துக்கள்" / "ச்" / "ச்" / "nested.png")          # duplicated nested folder
    _img(root / "மெய் எழுத்துக்கள்" / "வ்" / "வ்" / "வ்" / "deep.bmp")
    (root / "உயிர்மெய் எழுத்துக்கள" / "தௌ").mkdir(parents=True)           # empty class
    z = root / "உயிர்மெய் எழுத்துக்கள" / "கா"
    z.mkdir(parents=True)
    ok, buf = cv2.imencode(".png", np.zeros((10, 10), np.uint8))
    with zipfile.ZipFile(z / "imgs.zip", "w") as zf:
        zf.writestr("inner/x.png", buf.tobytes())
        zf.writestr("readme.txt", "no image")
    return root


def test_locate_root_and_nested_classes(tmp_path):
    root = _tree(tmp_path)
    found = locate_dataset_root(tmp_path / "scan", EXTS)
    assert found == root.resolve()
    classes, empty = discover_classes(found, EXTS)
    labels = sorted(c.label for c in classes)
    assert "ச்" in labels and "வ்" in labels and "அ" in labels
    nested = {c.label: c.nested for c in classes}
    assert nested["ச்"] and nested["வ்"] and not nested["அ"]
    assert {c.category for c in classes if c.label == "வ்"} == {"மெய் எழுத்துக்கள்"}


def test_empty_class_skipped(tmp_path):
    classes, empty = discover_classes(_tree(tmp_path).resolve(), EXTS)
    assert "தௌ" not in {c.label for c in classes}
    assert any(p.name == "தௌ" for p in empty)


def test_zip_extracted_outside_dataset(tmp_path):
    root = _tree(tmp_path).resolve()
    classes, _ = discover_classes(root, EXTS)
    out = tmp_path / "interim" / "extracted"
    n = extract_zips(classes, out, EXTS)
    assert n == 1
    ka = [c for c in classes if c.label == "கா"][0]
    assert ka.files and all(out in f.parents for f in ka.files)
    assert not any(p.suffix == ".png" for p in (root / "உயிர்மெய் எழுத்துக்கள" / "கா").rglob("*"))


def test_unicode_path_reading(tmp_path):
    p = tmp_path / "தமிழ் folder (1)" / "ஙௌ image.png"
    _img(p, val=10)
    g = read_image_gray(p)
    assert g.shape == (20, 20) and g.dtype == np.uint8 and g.min() == 10


def test_alpha_composited_on_white(tmp_path):
    p = tmp_path / "alpha.png"
    rgba = np.zeros((10, 10, 4), np.uint8)            # fully transparent black -> white
    rgba[2:4, 2:4] = (0, 0, 0, 255)
    cv2.imencode(".png", rgba)[1].tofile(str(p))
    g = read_image_gray(p)
    assert g[0, 0] == 255 and g[2, 2] == 0


def test_nfc_normalisation_and_variants():
    decomposed = "க" + "ொ"                   # கொ typed as e-sign + aa-sign
    assert tt.nfc(decomposed) == "கொ" and len(tt.nfc(decomposed)) == 2
    assert tt.nfc("கௌ") == "கௌ"              # au sign composes
    # e-sign + LLA is a different string from the au sign even after NFC: never merged
    assert tt.nfc("தெள") != tt.nfc("தௌ")
    assert tt.parse_label("தெள")["kind"] == "variant"
    assert unicodedata.is_normalized("NFC", tt.nfc("ஙௌ"))


def test_source_groups():
    assert source_group("IMG-20230101-WA0003.jpg") == "whatsapp"
    assert source_group("Screenshot 2023-01-01.png") == "screenshot"
    assert source_group("IMG_20230105_101010.jpg") == "img_date"
    assert source_group("PXL_2023.jpg") == "pxl"
    assert source_group("Scanned_doc.png") == "scanned"
    assert source_group("12 (3).png") == "numeric"
    assert source_group("ka_resized.png") == "resized"
