"""Tkinter drawing canvas: write a Tamil letter with the mouse, get the live top-5.

Buttons: Predict, Clear, "Enroll as new letter" (asks for the label, saves the drawing to
data/support_set/<label>/ and updates the prototypes immediately - no retraining).
Tamil labels in the list are rendered with Pillow + HarfBuzz shaping (tamil_text.render_text)
so they show as glyphs even where Tk's own text shaping is incomplete.

    python src/app_canvas.py [--backend onnx|torch]
    python src/app_canvas.py --selftest       # draws a synthetic stroke, predicts, enrols, exits
"""
from __future__ import annotations

import argparse
import logging
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import tamil_text as tt
from config import force_utf8, setup_logging
from predict import Recognizer
from support_registry import DEFAULT_MODELS, DEFAULT_SUPPORT, enroll

log = logging.getLogger("app_canvas")
CANVAS = 320
PEN = 14


class CanvasApp:
    def __init__(self, root: tk.Tk, rec: Recognizer, support_dir: Path) -> None:
        self.root, self.rec, self.support_dir = root, rec, support_dir
        root.title("Tamil letter recogniser (few-shot)")
        self.img = Image.new("L", (CANVAS, CANVAS), 255)
        self.draw = ImageDraw.Draw(self.img)
        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, rowspan=8, padx=8, pady=8)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._end_stroke())
        self.last: Optional[tuple] = None
        tk.Button(root, text="Predict", width=22, command=self.predict).grid(row=0, column=1, sticky="n")
        tk.Button(root, text="Clear", width=22, command=self.clear).grid(row=1, column=1, sticky="n")
        tk.Button(root, text="Enroll as new letter", width=22, command=self.enroll_dialog).grid(row=2, column=1, sticky="n")
        self.live = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="live prediction", variable=self.live).grid(row=3, column=1, sticky="n")
        self.result_frame = tk.Frame(root)
        self.result_frame.grid(row=4, column=1, rowspan=4, sticky="nw", padx=8)
        self.status = tk.Label(root, text=f"{len(rec.labels)} letters enrolled", anchor="w")
        self.status.grid(row=8, column=0, columnspan=2, sticky="we", padx=8)
        self._photos: List[ImageTk.PhotoImage] = []

    # -- drawing ----------------------------------------------------------------------------
    def _paint(self, e) -> None:
        if self.last is not None:
            self.canvas.create_line(*self.last, e.x, e.y, width=PEN, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last, (e.x, e.y)], fill=0, width=PEN)
            r = PEN // 2
            self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=0)
        self.last = (e.x, e.y)

    def _end_stroke(self) -> None:
        self.last = None
        if self.live.get():
            self.predict()

    def clear(self) -> None:
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=255)
        self._show([])

    # -- recognition ------------------------------------------------------------------------
    def gray(self) -> np.ndarray:
        return np.asarray(self.img, dtype=np.uint8)

    def predict(self):
        g = self.gray()
        if g.min() > 200:
            self._show([]); return None
        pr = self.rec.predict_gray(g)
        rows = [(f"{'' if pr.known else 'unknown?  '}", None)] + [(lab, c) for lab, c in pr.top]
        self._show(rows, pr.score)
        return pr

    def _show(self, rows, score: Optional[float] = None) -> None:
        for w in self.result_frame.winfo_children():
            w.destroy()
        self._photos.clear()
        for i, (lab, conf) in enumerate(rows):
            if conf is None:
                if lab:
                    tk.Label(self.result_frame, text=f"below threshold (score {score:.1f})", fg="red").grid(row=i, column=0, columnspan=2, sticky="w")
                continue
            ph = ImageTk.PhotoImage(tt.render_text(lab, 30, (0, 0, 0)))
            self._photos.append(ph)
            tk.Label(self.result_frame, image=ph).grid(row=i, column=0, sticky="w")
            tk.Label(self.result_frame, text=f"{100 * conf:5.1f}%").grid(row=i, column=1, sticky="e")

    def enroll_dialog(self) -> None:
        label = simpledialog.askstring("Enroll", "Label of the letter you drew (e.g. தௌ):", parent=self.root)
        if label:
            self.enroll_current(label)

    def enroll_current(self, label: str) -> None:
        g = self.gray()
        if g.min() > 200:
            self.status.config(text="nothing drawn"); return
        enroll(self.rec.model, label, [g], self.support_dir)
        self.rec.reload()
        self.status.config(text=f"enrolled {tt.nfc(label)} - {len(self.rec.labels)} letters")


def synthetic_letter(text: str, size: int = CANVAS) -> np.ndarray:
    """Font-rendered letter on a white canvas (used by --selftest)."""
    img = Image.new("L", (size, size), 255)
    patch = tt.render_text(text, int(size * 0.55), (0, 0, 0))
    img.paste(Image.new("L", patch.size, 0), ((size - patch.width) // 2, (size - patch.height) // 2), patch.split()[-1])
    return np.asarray(img)


def main() -> None:
    force_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--support-dir", default=str(DEFAULT_SUPPORT))
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    rec = Recognizer(Path(a.models), Path(a.support_dir), a.backend)
    root = tk.Tk()
    app = CanvasApp(root, rec, Path(a.support_dir))
    if a.selftest:
        g = synthetic_letter("க")
        app.img.paste(Image.fromarray(g))
        root.update()
        pr = app.predict()
        log.info("selftest prediction for a font-rendered க: %s top=%s", pr.label, pr.top[:3])
        root.update()
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
