from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logging.debug("Received file upload request")
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({'message': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'message': 'No selected file'}), 400

        logging.debug(f"Received file: {file.filename}")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        logging.debug(f"DataFrame head: \n{df.head()}")

        return jsonify({'message': 'File uploaded and read successfully'})
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
