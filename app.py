# app.py
import os
from flask import Flask, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

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
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        return send_file(file_path)

@app.route('/watermark_detector')
def watermark_detector():
    return render_template('watermark_detector.html')

# Route to handle image upload and watermark detection
@app.route('/detect_success', methods=['POST'])
def detect_success():
    if request.method == 'POST':
        f = request.files['file']
        # Save the uploaded file
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        # Redirect to the result page with the file name
        return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True)
