import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import folium
import os
import joblib

class TerrorismModel:
    def __init__(self):
        self.african_countries = [
            "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", 
            "Cameroon", "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo", 
            "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", 
            "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", 
            "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", 
            "Namibia", "Niger", "Nigeria", "Republic of the Congo", "Rwanda", "São Tomé and Príncipe", 
            "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", 
            "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
        ]
        self.selected_columns = [
            'iyear', 'imonth', 'iday', 'latitude', 'longitude', 'country',
            'country_txt', 'region', 'region_txt', 'provstate', 'city', 'multiple',
            'success', 'suicide', 'targtype1','targtype1_txt', 'weaptype1',
            'weaptype1_txt', 'gname', 'attacktype1_txt', 'attacktype1', 'target1',
            'natlty1', 'natlty1_txt', 'nkill', 'property', 'dbsource'
        ]

    def preprocess_data(self, df):
        africa_df = df[df['country_txt'].isin(self.african_countries)]
        attack_type_df = africa_df[self.selected_columns]
        attack_type_df = self.fill_null_values(attack_type_df)
        encoded_df = self.encode_categorical_data(attack_type_df)
        return encoded_df

    def fill_null_values(self, df):
        avg_lat_long = df.groupby('country_txt')[['latitude', 'longitude']].mean().reset_index()
        df = df.merge(avg_lat_long, on='country_txt', suffixes=('', '_avg'))
        df['latitude'].fillna(df['latitude_avg'], inplace=True)
        df['longitude'].fillna(df['longitude_avg'], inplace=True)
        df.drop(columns=['latitude_avg', 'longitude_avg'], inplace=True)

        avg_nkill = df.groupby('country_txt')['nkill'].mean().reset_index()
        df = df.merge(avg_nkill, on='country_txt', suffixes=('', '_avg'))
        df['nkill'].fillna(df['nkill_avg'], inplace=True)
        df.drop(columns=['nkill_avg'], inplace=True)

        df.dropna(subset=['provstate', 'city', 'target1'], inplace=True)
        df['natlty1'].fillna('Unknown', inplace=True)
        df['natlty1_txt'].fillna('Unknown', inplace=True)
        return df

    def encode_categorical_data(self, df):
        categorical_cols_high_cardinality = ['provstate', 'city', 'gname', 'target1', 'natlty1', 'natlty1_txt']
        for col in categorical_cols_high_cardinality:
            df[col + '_encoded'] = df[col].astype('category').cat.codes
        df.drop(columns=categorical_cols_high_cardinality, inplace=True)

        object_columns_to_drop = ['country_txt', 'region_txt', 'targtype1_txt', 'weaptype1_txt', 'attacktype1_txt']
        df.drop(columns=object_columns_to_drop, inplace=True)

        one_hot_encoded_dbsource = pd.get_dummies(df['dbsource'], prefix='dbsource')
        df = pd.concat([df, one_hot_encoded_dbsource], axis=1)
        df.drop(columns=['dbsource'], inplace=True)
        df.columns = df.columns.str.replace('dbsource_', '')
        return df

    def train_model(self, X_train, y_train):
        lgbm = LGBMClassifier(learning_rate=0.1, max_depth=15, num_leaves=40, verbose=-1)
        lgbm.fit(X_train, y_train)
        return lgbm

    def deploy_model(self, df, training=True):
        encoded_df = self.preprocess_data(df)

        classes_to_remove = [4, 5, 8] 
        filtered_df = encoded_df[~encoded_df['attacktype1'].isin(classes_to_remove)]

        X = filtered_df.drop(columns=['attacktype1'])
        y = filtered_df['attacktype1']

        if training or not os.path.exists('terrorism_model.pkl'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            trained_model = self.train_model(X_train, y_train)

            # Save the model
            joblib.dump(trained_model, 'terrorism_model.pkl')
        else:
            # Load the model if it's already trained
            trained_model = joblib.load('terrorism_model.pkl')
            X_test, y_test = X, y  # Use all data for prediction in this case

        y_pred = trained_model.predict(X_test)

        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Visualize predictions
        X_test['predicted_attacktype1'] = y_pred
        map_html = self.visualize_predictions(X_test)

        return trained_model, report, accuracy, map_html

    def visualize_predictions(self, df):
        africa_map = folium.Map(location=[1.650801, 10.267895], zoom_start=4)

        attack_type_labels = {
            1: 'Assassination',
            2: 'Armed Assault',
            3: 'Bombing/Explosion',
            6: 'Kidnapping',
            7: 'Facility/Infrastructure Attack',
            9: 'Unknown'
        }
        
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Attack Type: {attack_type_labels.get(row['predicted_attacktype1'], 'Other')}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(africa_map)

        return africa_map._repr_html_()
