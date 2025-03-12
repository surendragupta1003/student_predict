import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Secure MongoDB URI
MONGO_URI = "mongodb+srv://surendraphulvasi:Manu7752@cluster0.vlii4ff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client['student_performance']
    collection = db['student_performance_prediction']
    st.success("✅ MongoDB Connected Successfully!")

except Exception as e:
    st.error(f"❌ MongoDB Connection Failed: {e}")

# Load model, scaler, and label encoder
@st.cache_resource
def load_model():
    with open('student_final_linear_model.pkl', 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

model, scaler, le = load_model()

def preprocessing_input_data(data):
    try:
        data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    except ValueError:
        st.error("Unknown category for Extracurricular Activities. Allowed: Yes/No")
        return None
    
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    preprocessed_data = preprocessing_input_data(data)
    if preprocessed_data is None:
        return None  
    prediction = model.predict(preprocessed_data)
    return float(prediction)

def main():
    st.title("🎓 Student Performance Prediction")
    
    hours_studied = st.number_input("📖 Hours Studied", min_value=1, max_value=10, value=4, step=1)
    previous_score = st.number_input("📊 Previous Score", min_value=40, max_value=100, value=70, step=1)
    extracu_activity = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"])
    sleeping_hours = st.number_input("😴 Sleeping Hours", min_value=4, max_value=10, value=5, step=1)
    question_solved = st.number_input("✏️ Nos. of Question Paper Solved", min_value=0, max_value=100, value=5, step=1)

    if st.button("🔮 Predict Score"):
        data = {
            "Extracurricular Activities": extracu_activity,
            "Hours Studied": hours_studied,
            "Previous Scores": previous_score,
            "Sleep Hours": sleeping_hours,
            "Sample Question Papers Practiced": question_solved
        }

        prediction = predict_data(data)
        if prediction is not None:
            predicted_score = int(round(prediction))
            st.success(f"🎯 Your predicted score is **{predicted_score}**")

            # Convert NumPy types to standard Python types
            data["Predicted Score"] = predicted_score
            data = {key: int(value) if isinstance(value, np.integer) else value for key, value in data.items()}

            try:
                collection.insert_one(data)
                st.success("✅ Prediction saved successfully in MongoDB!")
            except Exception as e:
                st.error(f"❌ Error saving to database: {e}")

if __name__ == "__main__":
    main()
