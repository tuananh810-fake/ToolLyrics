from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from high_precision_word_aligner import run_alignment


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
UPLOADS_DIR = JOBS_DIR / "uploads"
OUTPUTS_DIR = JOBS_DIR / "outputs"
DEBUGS_DIR = JOBS_DIR / "debug"
ALLOWED_AUDIO_EXTENSIONS = {".mp3"}
ALLOWED_LRC_EXTENSIONS = {".lrc"}

for directory in (UPLOADS_DIR, OUTPUTS_DIR, DEBUGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("TOOLLYRIC_SECRET_KEY", "toollyric-dev-secret")


def has_allowed_extension(filename: str, allowed_extensions: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed_extensions


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process():
    audio_file = request.files.get("audio")
    lrc_file = request.files.get("lrc")
    song = request.form.get("song", "").strip() or None
    language = request.form.get("language", "").strip() or None
    model_name = request.form.get("model", "large-v3").strip() or "large-v3"

    if audio_file is None or not audio_file.filename:
        flash("Bạn cần chọn file MP3.", "error")
        return redirect(url_for("index"))
    if lrc_file is None or not lrc_file.filename:
        flash("Bạn cần chọn file LRC.", "error")
        return redirect(url_for("index"))
    if not has_allowed_extension(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
        flash("File audio phải là .mp3.", "error")
        return redirect(url_for("index"))
    if not has_allowed_extension(lrc_file.filename, ALLOWED_LRC_EXTENSIONS):
        flash("File lyric phải là .lrc.", "error")
        return redirect(url_for("index"))
    if model_name not in {"large-v3", "medium"}:
        flash("Model không hợp lệ.", "error")
        return redirect(url_for("index"))

    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOADS_DIR / job_id
    job_output_dir = OUTPUTS_DIR / job_id
    job_debug_dir = DEBUGS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    audio_name = secure_filename(audio_file.filename)
    lrc_name = secure_filename(lrc_file.filename)
    audio_path = job_upload_dir / audio_name
    lrc_path = job_upload_dir / lrc_name
    output_path = job_output_dir / "lyrics_pro.json"

    audio_file.save(audio_path)
    lrc_file.save(lrc_path)

    try:
        payload = run_alignment(
            audio_path,
            lrc_path,
            output_path,
            song=song,
            model_name=model_name,
            language=language,
            debug_dir=job_debug_dir,
        )
    except Exception as exc:
        flash(f"Xử lý thất bại: {exc}", "error")
        return redirect(url_for("index"))

    total_lines = len(payload.get("data", []))
    total_words = sum(len(line.get("words", [])) for line in payload.get("data", []))
    return render_template(
        "result.html",
        song_title=payload.get("song", audio_path.stem),
        job_id=job_id,
        output_filename=output_path.name,
        output_size_kb=max(1, round(output_path.stat().st_size / 1024)),
        total_lines=total_lines,
        total_words=total_words,
        model_name=model_name,
    )


@app.get("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    safe_filename = secure_filename(filename)
    file_path = OUTPUTS_DIR / job_id / safe_filename
    if not file_path.is_file():
        flash("Không tìm thấy file export.", "error")
        return redirect(url_for("index"))
    return send_file(file_path, as_attachment=True, download_name=safe_filename)


if __name__ == "__main__":
    host = os.environ.get("TOOLLYRIC_HOST", "127.0.0.1")
    port = int(os.environ.get("TOOLLYRIC_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
