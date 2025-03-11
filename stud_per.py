import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder



def load_model():
    with open(r'C:\Users\suren\Desktop\FSDS_Tutorial\ML\student_per_app\student_final_linear_model.pkl', 'rb') as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le


def preprocessing_input_data(data, scaler, le):
    data["Extracurricular Activities"]= le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler, le = load_model()
    preprocessed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocessed_data)
    return prediction


def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hours_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=4)
    previous_score =  st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    extracu_activity = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleeping_hours = st.number_input("Sleeping Hours", min_value=4, max_value=10, value=5)
    question_solved = st.number_input("Nos. of Question Paper Solved", min_value=0, max_value=100, value=5)
    
    if st.button("Predict_Score"):
        data = {
            "Extracurricular Activities": extracu_activity,
            "Hours Studied": hours_studied,
            "Previous Scores": previous_score,            
            "Sleep Hours": sleeping_hours,
            "Sample Question Papers Practiced": question_solved
        }
        prediction = predict_data(data)
        st.success(f"Your predicted score is {prediction}")
     
    
    
if __name__ == "__main__":
    main()