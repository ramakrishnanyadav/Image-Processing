from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import cv2
import numpy as np
import os
import uuid
from io import BytesIO
import zipfile
import time
from pathlib import Path

app = Flask(__name__)
app.secret_key = "secret_key_123"  # for flash messages

# Ensure processed folder exists
UPLOAD_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# IMAGE PROCESSING FUNCTIONS
# ===============================
def translation(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    return cv2.warpAffine(img, M, (cols, rows))

def rotation(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def scaling(img):
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

def brightness_enhance(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=50)

def brightness_suppress(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=-50)

def contrast(img):
    return cv2.convertScaleAbs(img, alpha=1.5, beta=0)

def hist_eq(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def negative(img):
    return 255 - img

def gray_slice_no_bg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[(gray > 100) & (gray < 200)] = 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def gray_slice_bg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = np.where((gray > 100) & (gray < 200), 255, gray)
    return cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * (np.log(img + 1))
    return np.array(log_img, dtype=np.uint8)

def power_law(img, gamma=0.5):
    normalized = img / 255.0
    out = np.power(normalized, gamma)
    return np.uint8(out * 255)

def frequency_response(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return cv2.cvtColor(np.uint8(magnitude / np.max(magnitude) * 255), cv2.COLOR_GRAY2BGR)

def zoom_pixel_replication(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# Filters
def box_filter(img):
    return cv2.blur(img, (5, 5))

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def median_filter(img):
    return cv2.medianBlur(img, 5)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

# ====================================
# HELPER FUNCTIONS
# ====================================
def clear_old_images(hours=1):
    now = time.time()
    cutoff = now - hours * 3600
    folder = Path(UPLOAD_FOLDER)
    for file in folder.glob("*"):
        if file.is_file() and file.stat().st_mtime < cutoff:
            file.unlink()

def reset_gallery():
    folder = Path(UPLOAD_FOLDER)
    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()

# ====================================
# ROUTES
# ====================================
@app.route('/')
def index():
    return render_template('index.html', processed_images=None)

@app.route('/process', methods=['POST'])
def process_images():
    reset_gallery()
    clear_old_images(hours=1)

    files = request.files.getlist('images')
    operation = request.form['operation']
    processed_paths = []

    if not files:
        flash("No images uploaded.", "error")
        return redirect(url_for("index"))

    for file in files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        op_func = {
            'translation': translation,
            'rotation': rotation,
            'scaling': scaling,
            'brightness_enhance': brightness_enhance,
            'brightness_suppress': brightness_suppress,
            'contrast': contrast,
            'hist_eq': hist_eq,
            'negative': negative,
            'gray_slice_no_bg': gray_slice_no_bg,
            'gray_slice_bg': gray_slice_bg,
            'log': log_transform,
            'power': power_law,
            'freq': frequency_response,
            'zoom': zoom_pixel_replication,
            'box_filter': box_filter,
            'gaussian_filter': gaussian_filter,
            'median_filter': median_filter,
            'bilateral_filter': bilateral_filter
        }.get(operation, None)

        out = op_func(img) if op_func else img
        out = np.array(out, dtype=np.uint8)
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        filename = f"processed_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, out)
        processed_paths.append(filename)

    flash(f"{len(processed_paths)} images processed successfully!", "success")
    return render_template('index.html', processed_images=processed_paths)

@app.route('/download_all')
def download_all():
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            zipf.write(filepath, arcname=filename)
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip',
                     as_attachment=True, download_name='processed_images.zip')

# ====================================
# RUN APP (Railway-ready)
# ====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
