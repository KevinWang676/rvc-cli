"""
app.py  –  minimal FastAPI wrapper around the RVC “infer” pipeline
-----------------------------------------------------------------
POST /voice‑convert
-------------------
JSON body:
{
    "pitch":        int  (‑24 … 24),
    "input_url":    "https://…/input.wav",
    "pth_url":      "https://…/model.pth",
    "index_url":    "https://…/model.index"
}

Returns: WAV file produced by RVC and deletes **all** temporary files
after the response has been streamed to the client.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, conint

import subprocess
import mimetypes

# ---- RVC import ------------------------------------------------------------
# The original script you pasted exposes `import_voice_converter`,
# which lazily builds a global VoiceConverter instance.
from rvc_cli import import_voice_converter  # rename if your CLI file uses a different name

converter = import_voice_converter()  # cache the singleton

# ---------------------------------------------------------------------------

app = FastAPI(title="RVC Voice‑Conversion API", version="1.0.0")


class VoiceConversionRequest(BaseModel):
    pitch: conint(ge=-24, le=24)  # validate pitch range
    input_url: HttpUrl
    pth_url:   HttpUrl
    index_url: HttpUrl


# --------------------- utility helpers -------------------------------------

def _ensure_wav(src: Path, work_dir: Path) -> Path:
    """
    If *src* is already a WAV, return it. Otherwise transcode to WAV with FFmpeg
    and return the new file path. Requires ffmpeg in $PATH.
    """
    if src.suffix.lower() == ".wav":
        return src

    dst = work_dir / f"{src.stem}_converted.wav"

    # basic safety – reject obviously non‑audio content
    ctype, _ = mimetypes.guess_type(src.name)
    if not (ctype or "").startswith("audio"):
        raise HTTPException(status_code=400, detail="Input is not an audio file")

    # ffmpeg -y -i in.xxx -ar 48000 -ac 2 out.wav
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "48000", "-ac", "2", str(dst)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise HTTPException(
            status_code=500,
            detail="Failed to convert audio; make sure FFmpeg is installed on the server.",
        )
    return dst

async def _download(url: str, dest: Path) -> None:
    """Stream a remote file to *dest*."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch {url}")
        dest.write_bytes(r.content)


def _voice_convert(
    wav_in: Path,
    wav_out: Path,
    pth_file: Path,
    index_file: Path,
    pitch: int,
) -> None:
    """Synchronous call to the RVC infer pipeline."""
    converter.convert_audio(
        audio_input_path=str(wav_in),
        audio_output_path=str(wav_out),
        model_path=str(pth_file),
        index_path=str(index_file),
        pitch=pitch,
        # reasonable defaults – tweak if you like
        filter_radius=3,
        index_rate=0.3,
        volume_envelope=1.0,
        protect=0.33,
        hop_length=128,
        f0_method="rmvpe",
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        clean_audio=False,
        clean_strength=0.7,
        export_format="WAV",
        embedder_model="contentvec",
        sid=0,
    )


# ---------------------------- endpoint -------------------------------------


@app.post("/voice-convert", response_class=FileResponse)
async def voice_convert(
    req: VoiceConversionRequest, background: BackgroundTasks
):
    # 1. create a private temp workspace
    tmp_dir = Path(tempfile.mkdtemp(prefix="rvc_"))
    background.add_task(shutil.rmtree, tmp_dir, ignore_errors=True)

    # 2. download user‑supplied files
    wav_in  = tmp_dir / Path(req.input_url.path).name
    pth     = tmp_dir / "model.pth"
    index   = tmp_dir / "model.index"

    try:
        await asyncio.gather(
            _download(str(req.input_url), wav_in),
            _download(str(req.pth_url),   pth),
            _download(str(req.index_url), index),
        )
    except HTTPException:
        # ensure the workspace is wiped even when downloads fail
        raise

    # 3. build output filename:  <original‑stem>_output.wav
    wav_source = _ensure_wav(wav_in, tmp_dir)
    out_wav    = tmp_dir / f"{wav_source.stem}_output.wav"

    # 4. run conversion in a worker thread so the event‑loop isn’t blocked
    try:
        await asyncio.to_thread(
            _voice_convert, wav_in, out_wav, pth, index, req.pitch
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # 5. stream the WAV back and schedule its deletion afterwards
    background.add_task(out_wav.unlink, missing_ok=True)
    background.add_task(wav_in.unlink,  missing_ok=True)
    background.add_task(pth.unlink,     missing_ok=True)
    background.add_task(index.unlink,   missing_ok=True)

    return FileResponse(
        path=out_wav,
        media_type="audio/wav",
        filename=f"{uuid.uuid4().hex}.wav",
        background=background,
    )
