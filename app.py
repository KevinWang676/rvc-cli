"""
app.py – unified FastAPI backend

* /voice‑convert   – RVC voice conversion
* /uvr-remove      – UVR vocal / instrumental separation

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import mimetypes
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, conint

# ─────────────── RVC IMPORT (lazy singleton) ────────────────────────────────
from rvc_cli import import_voice_converter  # change if your module is named differently

converter = import_voice_converter()

# ─────────────── FASTAPI APP ────────────────────────────────────────────────
app = FastAPI(
    title="Audio AI Backend",
    version="2.0.0",
    description="Voice conversion (RVC) + vocal removal (UVR)",
)

# ─────────────────────── COMMON HELPERS ─────────────────────────────────────


async def _download(url: str, dest: Path) -> None:
    """Stream *url* to *dest* (raises HTTPException on failure)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch {url}")
        dest.write_bytes(r.content)


def _ensure_wav(src: Path, work_dir: Path) -> Path:
    """If *src* isn’t WAV, transcode with FFmpeg → 48 kHz stereo WAV."""
    if src.suffix.lower() == ".wav":
        return src

    ctype, _ = mimetypes.guess_type(src.name)
    if not (ctype or "").startswith("audio"):
        raise HTTPException(status_code=400, detail="Input is not an audio file")

    dst = work_dir / f"{src.stem}_converted.wav"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "48000", "-ac", "2", str(dst)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise HTTPException(
            status_code=500,
            detail="FFmpeg failed or is missing on the server.",
        )
    return dst


# ──────────────────────── RVC VOICE‑CONVERT ─────────────────────────────────
class VoiceConversionRequest(BaseModel):
    pitch: conint(ge=-24, le=24)
    input_url: HttpUrl          # audio (wav/mp3/flac…)
    model_url: HttpUrl          # ZIP that holds *.pth & *.index

def _voice_convert(
    wav_in: Path, wav_out: Path, pth_file: Path, index_file: Path, pitch: int
) -> None:
    """Blocking call into the RVC VoiceConverter."""
    converter.convert_audio(
        audio_input_path=str(wav_in),
        audio_output_path=str(wav_out),
        model_path=str(pth_file),
        index_path=str(index_file),
        pitch=pitch,
        filter_radius=3,
        index_rate=0.3,
        volume_envelope=1.0,
        protect=0.33,
        hop_length=128,
        f0_method="rmvpe",
        split_audio=False,
        export_format="WAV",
        embedder_model="contentvec",
        sid=0,
    )


# ─── 2.  VOICE‑CONVERT ENDPOINT ─────────────────────────────────────────────
import zipfile, itertools
@app.post("/voice-convert", response_class=FileResponse)
async def voice_convert(req: VoiceConversionRequest, background: BackgroundTasks):
    tmp = Path(tempfile.mkdtemp(prefix="rvc_"))
    background.add_task(shutil.rmtree, tmp, ignore_errors=True)

    # 2‑a. download audio & model ZIP
    wav_src   = tmp / Path(req.input_url.path).name
    model_zip = tmp / "model.zip"
    await asyncio.gather(
        _download(str(req.input_url), wav_src),
        _download(str(req.model_url), model_zip),
    )

    # 2‑b. extract ZIP (nested folders ok)
    extract_dir = tmp / "model"
    extract_dir.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(model_zip) as zf:
            zf.extractall(extract_dir)

        # locate first *.pth and *.index anywhere in the tree
        pth_path   = next(itertools.chain(extract_dir.rglob("*.pth")),  None)
        index_path = next(itertools.chain(extract_dir.rglob("*.index")), None)
        if not pth_path or not index_path:
            raise HTTPException(status_code=400, detail="ZIP does not contain .pth and .index")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded model is not a valid ZIP")

    # 2‑c. make sure input is WAV
    wav_for_rvc = _ensure_wav(wav_src, tmp)
    out_wav     = tmp / f"{wav_for_rvc.stem}_output.wav"

    # 2‑d. run conversion in a worker thread
    try:
        await asyncio.to_thread(_voice_convert, wav_for_rvc, out_wav, pth_path, index_path, req.pitch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # 2‑e. schedule cleanup of every artefact
    for p in (wav_src, model_zip, out_wav, pth_path, index_path):
        background.add_task(p.unlink, missing_ok=True)
    background.add_task(shutil.rmtree, extract_dir, ignore_errors=True)

    return FileResponse(
        path=out_wav,
        media_type="audio/wav",
        filename=f"{uuid.uuid4().hex}.wav",
        background=background,
    )


# ────────────────────────── UVR VOCAL REMOVAL ───────────────────────────────
class UVRRequest(BaseModel):
    audio_url: HttpUrl
    model_filename: str  # e.g. "2_HP-UVR.pth"


# ──────────────── patched helper ───────────────────────────────────────────
def _uvr_separate(audio_path: Path, model_filename: str, out_dir: Path) -> list[Path]:
    from uvr.separator import Separator

    sep = Separator(
        model_file_dir="uvr/tmp/audio-separator-models/",
        output_dir=str(out_dir),
        output_format="MP3",
        normalization_threshold=0.9,
    )
    sep.load_model(model_filename=model_filename)

    raw_paths: list[str] = sep.separate(str(audio_path))

    # --- NEW: make sure every path is absolute & exists --------------------
    abs_paths: list[Path] = []
    for p in raw_paths:
        p_path = Path(p)
        if not p_path.is_absolute():
            p_path = out_dir / p_path          # <─ key fix
        p_path = p_path.resolve()
        if not p_path.exists():
            raise RuntimeError(f"UVR reported missing file: {p_path}")
        abs_paths.append(p_path)
    # ----------------------------------------------------------------------

    return abs_paths

@app.post("/uvr-remove", response_class=FileResponse)
async def uvr_remove(req: UVRRequest, background: BackgroundTasks):
    tmp = Path(tempfile.mkdtemp(prefix="uvr_"))
    background.add_task(shutil.rmtree, tmp, ignore_errors=True)

    src = tmp / Path(req.audio_url.path).name
    await _download(str(req.audio_url), src)

    try:
        stems = await asyncio.to_thread(_uvr_separate, src, req.model_filename, tmp)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UVR failed: {e}")

    if not stems:
        raise HTTPException(status_code=500, detail="No stems produced.")

    zip_path = tmp / f"{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in stems:
            zf.write(f, arcname=f.name)

    for p in stems + [src, zip_path]:
        background.add_task(p.unlink, missing_ok=True)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="uvr_output.zip",
        background=background,
    )
