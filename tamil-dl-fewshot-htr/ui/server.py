"""FastAPI app for the interactive UI. Thin: validation + HTTP; all ML lives in ``ui/service.py``."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from service import ImageError, LabelError, RecognitionService

log = logging.getLogger("ui.server")
STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/x-ms-bmp", "application/octet-stream", ""}


class SelectModel(BaseModel):
    id: str


def create_app(service: Optional[RecognitionService] = None, load_in_background: bool = True,
               public: Optional[bool] = None) -> FastAPI:
    """Build the app. The model loads in a background thread so /api/health answers at once.

    ``public`` (default: env ``TAMIL_UI_PUBLIC=1``) is for a shared internet deployment: enrolment and
    model switching change state for every visitor, so they are turned off.
    """
    if public is None:
        public = os.environ.get("TAMIL_UI_PUBLIC", "").lower() in ("1", "true", "yes")
    svc = service or RecognitionService()
    app = FastAPI(title="Tamil handwriting recogniser", docs_url="/api/docs", redoc_url=None)
    app.state.svc = svc

    def _load() -> None:
        try:
            svc.load()
        except Exception:
            log.exception("initial model load failed")

    if svc.rec is None:
        if load_in_background:
            threading.Thread(target=_load, name="model-load", daemon=True).start()
        else:
            _load()

    async def _read(upload: UploadFile) -> bytes:
        if (upload.content_type or "").split(";")[0].strip().lower() not in ALLOWED_TYPES:
            raise HTTPException(415, "Please upload a PNG, JPEG, WebP or BMP image.")
        data = await upload.read(svc.max_upload + 1)
        if len(data) > svc.max_upload:
            raise HTTPException(413, f"Image is larger than {svc.max_upload // (1024 * 1024)} MB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        return data

    def _guard(fn, *a, **kw) -> Any:
        try:
            return fn(*a, **kw)
        except ImageError as exc:
            raise HTTPException(400, str(exc).capitalize() + ".")
        except LabelError as exc:
            raise HTTPException(400, str(exc).capitalize() + ".")
        except RuntimeError as exc:
            raise HTTPException(503, "The model is still loading. Please try again in a moment.") if "not ready" in str(exc) \
                else HTTPException(500, "Something went wrong on the server.")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {**svc.health(), "public": public}

    @app.get("/api/classes")
    def classes() -> List[Dict[str, Any]]:
        if svc.rec is None:
            raise HTTPException(503, "The model is still loading.")
        return svc.classes()

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...), gradcam: bool = Query(False)) -> Dict[str, Any]:
        data = await _read(image)
        return _guard(svc.predict_bytes, data, gradcam)

    @app.post("/api/predict/stream")
    async def predict_stream(image: UploadFile = File(...), session: str = Form("default")) -> Dict[str, Any]:
        data = await _read(image)
        return _guard(svc.predict_stream, data, session)

    @app.post("/api/enroll")
    async def enroll(label: str = Form(...), images: List[UploadFile] = File(...)) -> Dict[str, Any]:
        if public:
            raise HTTPException(403, "Enrolling is turned off on the public demo. Run the app locally to teach new letters.")
        blobs = [await _read(u) for u in images]
        return _guard(svc.enroll_images, label, blobs)

    @app.post("/api/practice/check")
    async def practice(target: str = Form(...), image: UploadFile = File(...)) -> Dict[str, Any]:
        data = await _read(image)
        return _guard(svc.practice_check, target, data)

    @app.get("/api/models")
    def models() -> List[Dict[str, Any]]:
        return svc.models_info()

    @app.post("/api/models/select")
    def select(body: SelectModel) -> Dict[str, Any]:
        if public:
            raise HTTPException(403, "Model switching is turned off on the public demo.")
        try:
            svc.load(body.id)
        except KeyError:
            raise HTTPException(404, "Unknown model.")
        except Exception:
            raise HTTPException(500, "Could not load that model; the previous model is still active.")
        return {**svc.health(), "public": public}

    @app.exception_handler(Exception)
    async def _unhandled(_req, exc: Exception) -> JSONResponse:  # never leak raw tracebacks to the page
        log.exception("unhandled error: %s", exc)
        return JSONResponse({"detail": "Something went wrong on the server."}, status_code=500)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
