# app.py
import os
from flask import Flask, render_template, request, send_file,send_from_directory
from PIL import Image, ImageDraw, ImageFont
from embed import embed_watermark
from extractionModel import extract_signature_watermark, calculate_accuracy
from extractionModel import signature_watermark as fixed_sw
import numpy as np

app = Flask(__name__)
UPLOADS_DIR = 'flask-uploads'

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
        result_path = embed_watermark(file_path, UPLOADS_DIR)
        return send_file(result_path)
        # return render_template('embed_result.html', result_path=result_path)


@app.route('/watermark_extractor')
def watermark_extractor():
    return render_template('watermark_extractor.html', fixed_sw=str(fixed_sw).strip('[').strip(']'))

# Route to handle image upload and watermark detection


@app.route('/extract_success', methods=['POST'])
def extract_success():
    if request.method == 'POST':
        f = request.files['file']
        signature = request.form['signature']
        signature = np.fromstring(signature, dtype=int, sep=' ')
        file_path = os.path.join(UPLOADS_DIR, f.filename)
        f.save(file_path)

        extracted_sw = extract_signature_watermark(file_path)
        ncc_accuracy = calculate_accuracy(extracted_sw, fixed_sw)

        return render_template('extractor_result.html',
                               extracted_signature=extracted_sw,
                               input_signature=signature,
                               accuracy=ncc_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
