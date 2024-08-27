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
        
        # Initialize the model
        model = TerrorismModel()

        # Deploy the model (load or train)
        trained_model, report, accuracy, map_html = model.deploy_model(df, training=False)
        
        return render_template('index.html', report=report, accuracy=accuracy, map_html=map_html)


@app.route('/filter_map')
def filter_map():
    filter_type = request.args.get('filter')
    
    # Logic to filter your data based on the filter_type
    filtered_data = filter_data(filter_type)
    
    # Generate the map with filtered data
    folium_map = generate_map(filtered_data)
    map_html = folium_map._repr_html_()

    return map_html

def filter_data(filter_type):
    # Example logic to filter data
    if filter_type == "bombing":
        return [data for data in your_data if data['attack_type'] == 'Bombing']
    elif filter_type == "assassination":
        return [data for data in your_data if data['attack_type'] == 'Assassination']
    elif filter_type == "armed_assault":
        return [data for data in your_data if data['attack_type'] == 'Armed Assault']
    else:
        return df  # Show all data if filter is "all"

def generate_map(filtered_data):
    folium_map = folium.Map(location=[0, 0], zoom_start=2)
    for item in filtered_data:
        folium.Marker(
            location=[item['latitude'], item['longitude']],
            popup=item['attack_type']
        ).add_to(folium_map)
    return folium_map



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    #app.run(debug=True)
