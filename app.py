from flask import Flask, request, render_template
import pandas as pd
from terrorism_model import TerrorismModel

app = Flask(__name__)

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
