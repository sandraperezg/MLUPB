# -*- coding: utf-8 -*-
"""streamlitDespliegueClasificacion.ipynb


"""



import streamlit as st
import pandas as pd
import joblib
import os

# Function to preprocess the data
def preprocess_data(df, encoder, scaler):
    df['Felder'] = df['Felder'].astype('category')
    df = df.drop(['ID', 'Año - Semestre', 'Aprobo'], axis=1)

    # Apply One-Hot Encoding
    encoded_felder = encoder.transform(df[['Felder']])
    encoded_felder_df = pd.DataFrame(encoded_felder, columns=encoder.get_feature_names_out(['Felder']))
    df = pd.concat([df.drop('Felder', axis=1), encoded_felder_df], axis=1)

    # Apply Scaling
    df['Examen_admisión'] = scaler.transform(df[['Examen_admisión']]) # Use transform after fit_transform was called during training

    return df

try:
    encoder = joblib.load('onehot_encoder.pkl')
    scaler = joblib.load('standard_scaler.pkl')
    nnm_model = joblib.load('neural_network_model.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'saved_models' directory with required files exists.")
    st.stop()

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("Sube tu archivo Excel (solo .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Get sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

        # Add a selectbox for sheet selection
        selected_sheet = st.selectbox("Selecciona la hoja de cálculo:", sheet_names)

        # Read the selected sheet into a pandas DataFrame
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

        st.subheader("Datos cargados:")
        st.write(df.head())

        # Preprocess the data
        st.subheader("Datos preprocesados:")
        processed_df = preprocess_data(df.copy(), encoder, scaler)
        st.write(processed_df.head())

        # Make predictions using different models
        st.subheader("Predicciones:")

        # Random Forest Classifier
        rf_predictions = rf_model.predict(processed_df)
        df['Predicted_Aprobó_RF'] = rf_predictions
        st.write("Predicciones (Random Forest Classifier): ")
        st.write(df[['ID', 'Predicted_Aprobó_RF']].head())

        # Neural Network
        nn_predictions = nnm_model.predict(processed_df)
        df['Predicted_Aprobó_NN'] = nn_predictions
        st.write("Predicciones (Neural Network):")
        st.write(df[['ID', 'Predicted_Aprobó_NN']].head())

        # KNN
        knn_predictions =  knn_model.predict(processed_df)
        df['Predicted_Aprobó_KNN'] = knn_predictions
        st.write("Predicciones (knn):")
        st.write(df[['ID', 'Predicted_Aprobó_knn']].head())
        st.subheader("Resultados Completos:")
        
        # SVM
        svm_predictions = svm_model.predict(processed_df)
        df['Predicted_Aprobó_SVM'] = svm_predictions
        st.write("Predicciones (SVM):")
        st.write(df[['ID', 'Predicted_Aprobó_SVM']].head())


        
       st.subheader("Resultados Completos:")
        # Check if the columns exist before trying to display them
        display_cols = ['ID', 'Examen_admisión', 'Felder', 'Predicted_Aprobó_RF', 'Predicted_Aprobó_SVM']

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
