import os
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from classifficationModel import DeepANN  # Import your model

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 40  # Assuming 40 MFCC features
num_classes = 6  # Change based on your dataset
model = DeepANN(input_size, num_classes).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Preprocess audio file
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)  # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCCs
    mfcc = np.mean(mfcc.T, axis=0)  # Take mean over time
    return mfcc.reshape(1, -1)  # Reshape for model input


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "" or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Extract features
        features = extract_features(filepath)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            output = model(features_tensor)
            prediction = F.softmax(output, dim=1)
            predicted_label = torch.argmax(prediction, dim=1).item()

        return render_template("index.html", filename=filename, label=predicted_label)

    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
