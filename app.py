from flask import Flask, request, render_template
import pandas as pd
from terrorism_model import TerrorismModel
import requests
import os

app = Flask(__name__)

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    
    with open(destination, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)

# Download the GTD.csv file
file_id = '1z_lQyGMGm3KHY2BDJvzRkqhsU8hp6uE1'
destination = 'GTD.csv'

if not os.path.exists(destination):
    download_file_from_google_drive(file_id, destination)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        df = pd.read_csv(file, encoding='latin-1')
        
        # Initialize and deploy the model
        model = TerrorismModel()
        trained_model, report, accuracy, map_html = model.deploy_model(df)
        
        return render_template('index.html', report=report, accuracy=accuracy, map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
