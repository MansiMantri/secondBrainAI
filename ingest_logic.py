"""
Ingestion helpers for mRAG.

We convert supported inputs into images so the vision engine can generate
OCR-free "Visual Summaries".

Supported input types:
- PDF: render each page to a PNG
- Image: accept PNG/JPG/WebP (convert to PNG for consistency)
- Video: extract a small number of frames (uses `ffmpeg` or `imageio-ffmpeg`)
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from typing import List, Optional, Sequence, Union

import pypdfium2 as pdfium
from PIL import Image


def pdf_slides_to_images(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    dpi: int = 200,
    fmt: str = "png",
    page_start: int = 0,
    page_end: Optional[int] = None,
    scale: Optional[float] = None,
) -> List[Path]:
    """
    Convert each PDF page to an image (PNG by default) using pypdfium2.

    Returns:
        List of paths to rendered page images, ordered by page index.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg", "webp"}:
        raise ValueError("Unsupported fmt. Use one of: png, jpg/jpeg, webp.")

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        n_pages = len(pdf)
        if page_end is None:
            page_end = n_pages - 1

        if page_start < 0 or page_end < page_start or page_end >= n_pages:
            raise ValueError(
                f"Invalid page range: page_start={page_start}, page_end={page_end}, n_pages={n_pages}"
            )

        if scale is None:
            # PDF canvas units are typically 1/72 inch.
            scale = float(dpi) / 72.0

        rendered_paths: List[Path] = []
        for i in range(page_start, page_end + 1):
            page = pdf.get_page(i)
            try:
                bitmap = page.render(
                    scale=scale,
                    rotation=0,
                    crop=(0, 0, 0, 0),
                    # Keep this minimal; this pipeline is for diagrams.
                    may_draw_forms=False,
                )
                try:
                    img: Image.Image = bitmap.to_pil()
                    out_path = output_dir / f"slide_{i + 1:03d}.{fmt}"
                    save_kwargs = {}
                    if fmt in {"jpg", "jpeg"}:
                        save_kwargs["quality"] = 95
                    img.save(out_path, **save_kwargs)
                    rendered_paths.append(out_path)
                finally:
                    # Free bitmap buffer immediately (memory management for large PDFs).
                    if hasattr(bitmap, "close"):
                        bitmap.close()
            finally:
                if hasattr(page, "close"):
                    page.close()

        return rendered_paths
    finally:
        if hasattr(pdf, "close"):
            pdf.close()


def extract_pdf_page_text(pdf_path: Union[str, Path], page_index: int) -> str:
    """
    Extract embedded text from one PDF page via PDFium (the PDF text layer).

    Returns \"\" for non-PDFs, out-of-range pages, or pages with no extractable text
    (e.g. scanned images only). This is the most reliable \"what the document says\"
    signal for text-based PDFs; vision summaries alone often hallucinate.
    """
    pdf_path = _as_path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        return ""

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if page_index < 0 or page_index >= len(pdf):
            return ""
        page = pdf.get_page(page_index)
        try:
            textpage = page.get_textpage()
            try:
                raw = textpage.get_text_range()
                return raw.replace("\r\n", "\n").replace("\r", "\n").strip()
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        pdf.close()


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def image_to_png(image_path: Union[str, Path], output_dir: Union[str, Path]) -> List[Path]:
    """
    Convert an image to PNG (single-image list for API consistency).
    """
    image_path = _as_path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {suffix}")

    out_path = output_dir / f"{image_path.stem}.png"
    if out_path.exists():
        return [out_path]

    img = Image.open(str(image_path)).convert("RGB")
    img.save(out_path, format="PNG")
    return [out_path]


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _get_ffmpeg_exe() -> Optional[str]:
    """Return system ffmpeg path, or bundled imageio-ffmpeg path if available."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    fmt: str = "wav",
    sample_rate: int = 16000,
) -> Path:
    """
    Extract the audio track from a video file to a WAV (or other) file.
    Uses system ffmpeg or imageio-ffmpeg. Required for transcription.
    """
    video_path = _as_path(video_path)
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Not a supported video path: {video_path.suffix}")

    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError(
            "Audio extraction requires ffmpeg. Install system ffmpeg (e.g. brew install ffmpeg) "
            "or pip install imageio-ffmpeg."
        )

    out_path = output_dir / f"{video_path.stem}_audio.{fmt}"
    # WAV: pcm_s16le, 16kHz mono for Whisper-friendly input
    if fmt.lower() == "wav":
        cmd = [
            ffmpeg_exe,
            "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(out_path),
        ]
    else:
        cmd = [
            ffmpeg_exe,
            "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vn",
            str(out_path),
        ]
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        raise RuntimeError("ffmpeg did not produce an output file.")
    return out_path


def _extract_video_frames_with_imageio(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_frames: int = 8,
) -> List[Path]:
    """
    Extract frames using `imageio` + `imageio-ffmpeg` (no system ffmpeg required).
    """
    try:
        import imageio  # type: ignore
        import imageio_ffmpeg  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Video ingestion requires `ffmpeg`/`ffprobe` or Python packages "
            "`imageio` + `imageio-ffmpeg`. Install with: "
            "`pip install imageio imageio-ffmpeg`."
        ) from e

    video_path = _as_path(video_path)
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = None
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None

    # Ensure imageio uses the bundled ffmpeg binary (if available).
    if ffmpeg_exe:
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg_exe)

    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        duration_s = meta.get("duration")

        # Try to get a reliable frame count.
        nframes: Optional[int] = None
        try:
            nframes = reader.count_frames()
            if isinstance(nframes, float) and nframes != float("inf"):
                nframes = int(nframes)
        except Exception:
            nframes = None

        indices: list[int] = []
        if isinstance(nframes, int) and nframes > 0:
            if max_frames <= 1:
                indices = [0]
            else:
                indices = [
                    int(round(i * (nframes - 1) / (max_frames - 1))) for i in range(max_frames)
                ]
        elif fps and duration_s:
            total_est = int(round(float(fps) * float(duration_s)))
            if total_est > 0:
                if max_frames <= 1:
                    indices = [0]
                else:
                    indices = [
                        int(round(i * (total_est - 1) / (max_frames - 1)))
                        for i in range(max_frames)
                    ]
        else:
            # Worst-case fallback: iterate sequentially and take evenly spaced samples
            # based on how many frames we've consumed so far.
            indices = []
            for idx, _ in enumerate(reader):
                if len(indices) >= max_frames:
                    break
                if max_frames == 0:
                    break
                # Simple heuristic: sample every N frames after seeing at least once.
                if idx == 0 or (idx % max(1, (max_frames * 5))) == 0:
                    indices.append(idx)

        if not indices:
            raise RuntimeError("No frames were extracted with imageio.")

        # Clamp indices and keep unique, sorted order.
        clean_indices: list[int] = []
        seen: set[int] = set()
        for idx in indices:
            idx = max(0, int(idx))
            if idx not in seen:
                clean_indices.append(idx)
                seen.add(idx)

        frame_paths: List[Path] = []
        for i, frame_idx in enumerate(clean_indices[:max_frames]):
            frame = reader.get_data(frame_idx)
            img = Image.fromarray(frame).convert("RGB")
            out_path = output_dir / f"frame_{i:04d}.png"
            img.save(out_path, format="PNG")
            frame_paths.append(out_path)

        if not frame_paths:
            raise RuntimeError("imageio produced no frames.")

        return frame_paths
    finally:
        try:
            reader.close()
        except Exception:
            pass


def extract_video_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_frames: int = 8,
    min_seconds_between_frames: float = 0.5,
) -> List[Path]:
    """
    Extract a small number of frames from a video into PNGs.

    Uses `ffprobe` to estimate duration and `ffmpeg` to sample frames.
    """
    video_path = _as_path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video extension: {video_path.suffix}")

    if not _ffmpeg_available():
        # Fall back to a pure-Python approach with bundled ffmpeg.
        return _extract_video_frames_with_imageio(
            video_path,
            output_dir,
            max_frames=max_frames,
        )

    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe duration in seconds.
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    duration_s: Optional[float] = None
    try:
        out = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT, text=True).strip()
        duration_s = float(out) if out else None
    except Exception:
        duration_s = None

    if not duration_s or duration_s <= 0:
        # Fallback: ~1 frame every 5 seconds.
        seconds_between = max(min_seconds_between_frames, 5.0)
    else:
        # Aim for `max_frames` roughly evenly spaced.
        seconds_between = max(min_seconds_between_frames, duration_s / float(max_frames))

    # Sample 1 frame per `seconds_between` seconds, capped by -frames:v.
    # fps filter requires the "fps=" prefix (e.g. fps=1/5), not a bare fraction.
    fps_filter = f"fps=1/{seconds_between:.6f}"

    out_pattern = str(output_dir / "frame_%04d.png")
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        fps_filter,
        "-frames:v",
        str(max_frames),
        "-start_number",
        "0",
        out_pattern,
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("ffmpeg produced no frames. Try increasing max_frames.")
    return frame_paths


def media_to_images(
    media_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    pdf_dpi: int = 250,
    video_max_frames: int = 8,
) -> List[Path]:
    """
    Convert an input (PDF/image/video) into a list of images.

    Returns:
        Sorted list of PNG paths ready for `visual_summary_from_image()`.
    """
    media_path = _as_path(media_path)
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = media_path.suffix.lower()
    if suffix == ".pdf":
        # Keep PDF pages in the same output_dir.
        return pdf_slides_to_images(media_path, output_dir, dpi=pdf_dpi, fmt="png")
    if suffix in IMAGE_EXTS:
        # Convert/copy into output_dir.
        return image_to_png(media_path, output_dir)
    if suffix in VIDEO_EXTS:
        return extract_video_frames(
            media_path,
            output_dir,
            max_frames=video_max_frames,
        )

    raise ValueError(f"Unsupported input type: {suffix}. Expected pdf/images/videos.")
