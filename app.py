# app.py
from flask import Flask, render_template, request
import os
import json
from utils.parser import parse_metadata_file
from vlm_model import generate_captions

UPLOAD_FOLDER = 'static/uploads'
IMG_FOLDER = 'img_folder'
OUTPUT_FOLDER = 'output_folder'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_TEXT_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_file = request.files.get('image')
    metadata_file = request.files.get('metadata')

    if image_file and allowed_file(image_file.filename, ALLOWED_IMAGE_EXTENSIONS) and \
       metadata_file and allowed_file(metadata_file.filename, ALLOWED_TEXT_EXTENSIONS):

        # Save uploaded files
        image_filename = image_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_file.filename)
        image_file.save(image_path)
        metadata_file.save(metadata_path)

        # Copy image to img_folder
        img_folder_path = os.path.join(IMG_FOLDER, image_filename)
        with open(image_path, 'rb') as src, open(img_folder_path, 'wb') as dst:
            dst.write(src.read())

        # Parse metadata
        metadata = parse_metadata_file(metadata_path)
        context_text = " ".join([
            metadata.get("section_header", ""),
            metadata.get("above_text", ""),
            metadata.get("caption", ""),
            metadata.get("below_text", ""),
            metadata.get("footnote", "")
        ])

        # Generate captions
        captions = generate_captions(image_path, context_text)

        # Save captions to JSON
        captions_data = {
            "image": image_filename,
            "concise_caption": captions['concise'],
            "detailed_caption": captions['detailed'],
            "confidence_scores": captions['confidence_scores']
        }
        json_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(image_filename)[0]}_captions.json")
        with open(json_path, 'w') as f:
            json.dump(captions_data, f, indent=4)

        # Return the result with colored captions
        return f"""
        <h3>Upload Successful!</h3>
        <img src='/{image_path}' width='300'><br><br>
        <b style='color: blue;'>Concise Caption:</b> {captions['concise']}<br>
        <b style='color: red;'>Detailed Caption:</b> {captions['detailed']}<br>
        <b>Confidence Scores:</b> {captions['confidence_scores']}<br><br>
        <a href="/">Go Back</a>
        """
    else:
        return "Invalid files! Please upload a valid image and .txt file."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)