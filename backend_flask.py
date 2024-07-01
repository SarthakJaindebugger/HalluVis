

from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import os
from speechbrain.inference.classifiers import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
import torchaudio
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import logging
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from flask_cors import CORS
#import ffprobe



app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    encoder_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models3/spkrec-xvect-voxceleb"
    )
    logger.info("Pretrained encoder classifier loaded successfully.")
except Exception as e:
    logger.error(f"Error loading pretrained encoder classifier: {e}")
    raise e

your_model_path = "/mnt/c/Users/shubh/OneDrive/Desktop/ML Projects/sarthak_new_flask/lstm_xvector.h5"

try:
    model = tf.keras.models.load_model(your_model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise e

# Variable to store the latest prediction label
latest_prediction = ""


def make_predictions(xvector_features):
    try:
        logger.info("Making predictions...")
        xvector_features = np.expand_dims(xvector_features, axis=0)
        xvector_features = np.expand_dims(xvector_features, axis=1)
        output = model.predict(xvector_features)
        predicted_class = np.argmax(output, axis=1)[0]

        predicted_label = "Violence Audio" if predicted_class == 1 else "Non-violence Audio"
        logger.info(f"Predicted label: {predicted_label}")
        return predicted_label

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


def extract_xvector_features(path, target_sample_rate=16000):
    try:
        logger.info(f"Processing audio file: {path}")

        signal, original_sample_rate = torchaudio.load(path)
        logger.info(f"Original sample rate: {original_sample_rate}")

        if original_sample_rate != target_sample_rate:
            logger.info("Resampling audio...")
            resample = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate)
            signal = resample(signal)

        if signal.shape[0] > 1:
            logger.info("Converting stereo to mono...")
            signal = signal.mean(dim=0, keepdim=True)

        logger.info("Extracting X-vector embeddings...")
        embeddings = encoder_classifier.encode_batch(signal)
        xvector_features = embeddings.mean(dim=1).squeeze().numpy()

        logger.info("X-vector extraction successful.")
        return xvector_features

    except Exception as e:
        logger.error(f"Ignoring {path} because audio is of 0 seconds or an error occurred: {e}")
        return None


def convert_to_wav(input_path, output_path):
    try:
        logger.info(f"Converting to WAV format: {input_path} -> {output_path}")

        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format='wav')

        logger.info("Conversion to WAV successful.")
        return True

    except Exception as e:
        logger.error(f"Error converting to WAV: {e}")
        return False


@app.route('/')
def home():
    try:
        # Render the HTML template with the welcome message and latest prediction
        template = """
        <html>
        <body>
            <h1>Welcome to the Violence Detection API!</h1>
            <p>Latest Prediction: {{ latest_prediction }}</p>

            <script>
                // Refresh the page every second
                setInterval(function(){
                    location.reload();
                }, 1000);
            </script>
        </body>
        </html>
        """

        return render_template_string(template, latest_prediction=latest_prediction)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error.'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        global latest_prediction

        # Check if the request has the file part
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'error': 'No selected audio file'}), 400

        logger.info(f"Received request for prediction with audio file: {audio_file.filename}")

        # Save the uploaded file with an absolute path
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
        audio_file.save(audio_path)

        # Check file format and convert to WAV if necessary
        if not filename.endswith('.wav'):
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_filepath = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), wav_filename)

            if convert_to_wav(audio_path, wav_filepath):
                os.remove(audio_path)  # Delete the original file
                audio_path = wav_filepath
            else:
                return jsonify({'error': 'Error converting audio to WAV format.'}), 400

        xvector_features = extract_xvector_features(audio_path)

        if xvector_features is not None:
            predicted_label = make_predictions(xvector_features)
            latest_prediction = predicted_label
            return jsonify({'predicted_label': predicted_label})
        else:
            return jsonify({'error': 'Feature extraction failed.'}), 400

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error.'}), 500


if __name__ == '__main__':
    app.run(debug=True)


