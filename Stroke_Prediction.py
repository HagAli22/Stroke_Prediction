import Tasks_Machine.Stroke_Prediction.Stroke_Prediction as st
import numpy as np
import pickle
import joblib

# Loading scaleres and model
def load_resources():   

    model = joblib.load("stroke_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model,scaler

def main():

    # title and header
    st.title("Stroke Prediction App")
    
    # Load resources
    model,scaler=load_resources()

    # Input interface
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
    ever_married = st.selectbox("Are you married?", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=22.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "smokes", "formerly smoked", "Unknown"])

    # Encoding categorical variables
    gender_map = {"Male": 1, "Female": 0}
    hypertension_map = {"No": 0, "Yes": 1}
    heart_disease_map = {"No": 0, "Yes": 1}
    ever_married_map = {"No": 0, "Yes": 1}
    work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
    smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

    # Make OneHotEconder for 
    rural = 1 if residence_type == "Rural" else 0
    urban = 1 if residence_type == "Urban" else 0

    # Combine input data 
    input_data = np.array([[
        gender_map[gender],
        age,
        hypertension_map[hypertension],
        heart_disease_map[heart_disease],
        ever_married_map[ever_married],
        work_type_map[work_type],
        avg_glucose_level,
        bmi,
        smoking_map[smoking_status],
        rural,
        urban
    ]])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_input)
        if prediction[0] == 0:
            st.success("✅ Low risk of stroke")

            
        else:
            st.error("⚠️ High risk of stroke")

# Run the app
if __name__ == '__main__':
    main()