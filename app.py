# app.py
import os
from flask import Flask, render_template, request, send_file,send_from_directory
from PIL import Image, ImageDraw, ImageFont
from embed import embed_watermark

app = Flask(__name__)
UPLOADS_DIR = '/home/advait/lwt-and-svd-watermarking/embedded-train-advay/uploads/'

# Route to serve the image from the uploads directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)
# Route to render the main page
@app.route('/')
def index():
    return render_template('base.html')

# Route to render the Watermark embedder page
@app.route('/watermark_embedder')
def watermark_embedder():
    return render_template('watermark_embedder.html')

@app.route('/embed_success', methods=['POST'])
def embed_success():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOADS_DIR, f.filename)
        f.save(file_path)
        # embed watermark
        result_path = embed_watermark(file_path)

        return send_file(result_path)

# Route to handle image upload and watermark detection
@app.route('/detect_success', methods=['POST'])
def detect_success():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True)
