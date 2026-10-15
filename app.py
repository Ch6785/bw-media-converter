from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv
from email.message import EmailMessage
import cv2
import numpy as np
import os
import shutil
import subprocess
import tempfile
import uuid
import smtplib
import ssl

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_IMAGES = {"png", "jpg", "jpeg", "bmp", "webp"}
ALLOWED_VIDEOS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}

# -------------------- DYNAMIC FFMPEG RESOLUTION --------------------
def get_binary_path(binary_name):
    # 1. Check system PATH first (Linux server / Render / Docker / System PATH)
    system_path = shutil.which(binary_name)
    if system_path:
        return system_path

    # 2. Local Windows fallback path
    win_ffmpeg = (
        r"C:\Users\chari\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
    )
    win_ffprobe = os.path.join(os.path.dirname(win_ffmpeg), "ffprobe.exe")

    if binary_name.startswith("ffmpeg") and os.path.isfile(win_ffmpeg):
        return win_ffmpeg
    elif binary_name.startswith("ffprobe") and os.path.isfile(win_ffprobe):
        return win_ffprobe

    return binary_name

FFMPEG = get_binary_path("ffmpeg")
FFPROBE = get_binary_path("ffprobe")


# -------------------- HELPER FUNCTIONS --------------------

def format_size(size):
    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def encode_image(image, fmt, quality):
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    else:
        params = []

    success, encoded = cv2.imencode("." + fmt, image, params)
    return encoded.tobytes() if success else None


def compress_image(image, target_size, fmt, priority):
    if fmt == "png":
        settings = range(10)
    elif priority == "quality":
        settings = range(100, 69, -5)
    elif priority == "size":
        settings = range(75, 19, -5)
    else:
        settings = range(95, 39, -5)

    best_data = None
    best_difference = float("inf")

    for setting in settings:
        data = encode_image(image, fmt, setting)

        if data is not None:
            difference = abs(len(data) - target_size)

            if difference < best_difference:
                best_data = data
                best_difference = difference

    return best_data


def process_image(data, extension, operation, reduction,
                  output_format, priority):
    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Could not read this image.")

    if operation == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        success, encoded = cv2.imencode(".png", gray)

        if not success:
            raise ValueError("Image conversion failed.")

        return encoded.tobytes(), "_grayscale.png"

    if output_format == "original":
        fmt = "jpg" if extension in ("jpg", "jpeg") else extension
    else:
        fmt = {"jpeg": "jpg"}.get(output_format, output_format)

    if fmt not in ("jpg", "png", "webp"):
        raise ValueError(
            "Choose JPEG, PNG, or WebP for image compression."
        )

    target_size = max(
        1,
        int(len(data) * (1 - reduction / 100))
    )

    output = compress_image(image, target_size, fmt, priority)

    if output is None:
        raise ValueError("Image compression failed.")

    return output, "_compressed." + fmt


def video_duration(path):
    result = subprocess.run(
        [
            FFPROBE,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        raise ValueError("Could not read the video duration.")

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        raise ValueError("Could not determine the video duration.")

    if duration <= 0:
        raise ValueError("Invalid video duration.")

    return duration


def process_video(input_path, output_path, operation, reduction):
    duration = video_duration(input_path)

    command = [
        FFMPEG,
        "-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "0:a?",
    ]

    if operation == "video_grayscale":
        command += [
            "-vf", "format=gray",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
        ]

    else:
        target_bytes = max(
            1,
            int(
                os.path.getsize(input_path)
                * (1 - reduction / 100)
            )
        )

        total_bitrate = int(target_bytes * 8 / duration)
        video_bitrate = max(100_000, total_bitrate - 96_000)

        command += [
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", str(video_bitrate),
            "-maxrate", str(video_bitrate),
            "-bufsize", str(video_bitrate * 2),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
        ]

    command += [
        "-movflags", "+faststart",
        output_path
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800
        )
    except subprocess.TimeoutExpired:
        raise ValueError(
            "Video processing timed out. Try a shorter video."
        )

    if result.returncode != 0 or not os.path.isfile(output_path):
        print("FFmpeg error:", result.stderr[-2000:])
        raise ValueError(
            "Video processing failed. Check that the video is valid."
        )


def safe_output_path(path):
    """Accept only files directly inside the outputs directory."""
    if not path:
        return None

    output_root = os.path.abspath(OUTPUT_DIR)
    candidate = os.path.abspath(path)

    if os.path.dirname(candidate) != output_root:
        return None

    return candidate


# -------------------- HOME PAGE --------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------- PROCESS ONE FILE --------------------

@app.route("/convert", methods=["POST"])
def convert():
    uploaded = request.files.get("media")

    if not uploaded or not uploaded.filename:
        return "Please select one file.", 400

    filename = os.path.basename(uploaded.filename)

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename else ""
    )

    original = uploaded.read()

    media_type = request.form.get("media_type", "image")
    operation = request.form.get("operation", "grayscale")
    output_format = request.form.get("output_format", "original")
    priority = request.form.get("priority", "balanced")

    try:
        reduction = int(request.form.get("reduction", "50"))
    except ValueError:
        return "Invalid reduction setting.", 400

    if reduction not in (25, 50, 75):
        return "Choose a 25%, 50%, or 75% reduction target.", 400

    if not original:
        return "The selected file is empty.", 400

    if media_type == "image" and len(original) > 50 * 1024 * 1024:
        return "Please choose an image smaller than 50 MB.", 400

    output_path = None

    try:
        if media_type == "image":
            if extension not in ALLOWED_IMAGES:
                raise ValueError("Please upload a supported image.")

            if operation not in ("grayscale", "compress"):
                raise ValueError("Choose a valid image operation.")

            output, suffix = process_image(
                original,
                extension,
                operation,
                reduction,
                output_format,
                priority
            )

            output_name = os.path.splitext(filename)[0] + suffix
            stored_name = uuid.uuid4().hex + "_" + output_name
            output_path = os.path.join(OUTPUT_DIR, stored_name)

            with open(output_path, "wb") as file:
                file.write(output)

        elif media_type == "video":
            if extension not in ALLOWED_VIDEOS:
                raise ValueError("Please upload a supported video.")

            if operation not in ("video_grayscale", "video_compress"):
                raise ValueError("Choose a valid video operation.")

            if not shutil.which(FFMPEG) and not os.path.isfile(FFMPEG):
                raise ValueError(
                    "FFmpeg was not found on the server. Check path settings."
                )

            suffix = (
                "_grayscale.mp4"
                if operation == "video_grayscale"
                else "_compressed.mp4"
            )

            output_name = os.path.splitext(filename)[0] + suffix
            stored_name = uuid.uuid4().hex + "_" + output_name
            output_path = os.path.join(OUTPUT_DIR, stored_name)

            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(
                    temp_dir,
                    "input." + extension
                )

                with open(input_path, "wb") as file:
                    file.write(original)

                process_video(
                    input_path,
                    output_path,
                    operation,
                    reduction
                )

        else:
            raise ValueError("Choose Images or Videos.")

        old_size = len(original)
        new_size = os.path.getsize(output_path)
        difference = old_size - new_size

        result = {
            "filename": filename,
            "output_name": output_name,
            "stored_name": stored_name,
            "output_path": output_path,
            "original_size": format_size(old_size),
            "output_size": format_size(new_size),
            "saved_size": format_size(abs(difference)),
            "saved_percent": round(
                difference / old_size * 100, 2
            ),
        }

        return render_template("result.html", result=result)

    except (ValueError, OSError, subprocess.SubprocessError) as error:
        if output_path and os.path.isfile(output_path):
            os.remove(output_path)

        return render_template("error.html", error=str(error)), 400


# -------------------- DOWNLOAD PROCESSED FILE --------------------

@app.route("/download")
def download():
    path = safe_output_path(request.args.get("path", ""))

    if not path:
        return "Invalid output file.", 400

    if not os.path.isfile(path):
        return "Output file unavailable. Please process it again.", 404

    return send_file(
        path,
        as_attachment=True,
        download_name=os.path.basename(path)
    )


# -------------------- EMAIL PROCESSED FILE --------------------

@app.route("/send-email", methods=["POST"])
def send_email():
    recipient = request.form.get("recipient", "").strip()
    path = safe_output_path(request.form.get("output_path", ""))

    if (
        not recipient
        or recipient.count("@") != 1
        or recipient.startswith("@")
        or recipient.endswith("@")
        or "." not in recipient.rsplit("@", 1)[-1]
    ):
        return render_template(
            "error.html",
            error="Please enter a valid recipient email address."
        ), 400

    if not path or not os.path.isfile(path):
        return render_template(
            "error.html",
            error="Processed file not found. Please process it again."
        ), 404

    sender = os.getenv("SENDER_EMAIL")
    app_password = os.getenv("SENDER_APP_PASSWORD")

    if not sender or not app_password:
        return render_template(
            "error.html",
            error="Email settings are missing. Check environment variables."
        ), 500

    if os.path.getsize(path) > 18 * 1024 * 1024:
        return render_template(
            "error.html",
            error="This file is too large to email reliably. Try a smaller file."
        ), 400

    message = EmailMessage()
    message["Subject"] = "Your processed media file"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        "Your processed media file is attached.\n\n"
        "Sent by B&W Media Converter."
    )

    extension = os.path.splitext(path)[1].lower()

    mime_types = {
        ".jpg": ("image", "jpeg"),
        ".jpeg": ("image", "jpeg"),
        ".png": ("image", "png"),
        ".webp": ("image", "webp"),
        ".mp4": ("video", "mp4"),
        ".mov": ("video", "quicktime"),
        ".avi": ("video", "x-msvideo"),
        ".webm": ("video", "webm"),
        ".mkv": ("video", "x-matroska"),
        ".m4v": ("video", "mp4"),
    }

    maintype, subtype = mime_types.get(
        extension,
        ("application", "octet-stream")
    )

    with open(path, "rb") as file:
        message.add_attachment(
            file.read(),
            maintype=maintype,
            subtype=subtype,
            filename=os.path.basename(path)
        )

    try:
        context = ssl.create_default_context()

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=30
        ) as server:
            server.starttls(context=context)
            server.login(sender, app_password)
            server.send_message(message)

    except (smtplib.SMTPException, OSError) as error:
        print("Email delivery failed:", error)

        return render_template(
            "error.html",
            error=(
                "Email could not be sent. Check SMTP access and app password."
            )
        ), 500

    return render_template(
        "email_sent.html",
        filename=os.path.basename(path),
        recipient=recipient
    )


# -------------------- START APP --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)