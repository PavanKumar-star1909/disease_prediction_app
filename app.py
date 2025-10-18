# app.py

import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Trained Models and Scalers ------------------
liver_model = joblib.load('models/liver_model.pkl')
kidney_model = joblib.load('models/kidney_model.pkl')
parkinsons_model = joblib.load('models/parkinsons_model.pkl')

liver_scaler = joblib.load('models/liver_scaler.pkl')
kidney_scaler = joblib.load('models/kidney_scaler.pkl')
parkinsons_scaler = joblib.load('models/parkinsons_scaler.pkl')

st.title("Multiple Disease Prediction System")

# ------------------ Liver Disease Prediction ------------------
st.header("Liver Disease Prediction")

age = st.number_input("Age", value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin", value=1.0)
direct_bilirubin = st.number_input("Direct Bilirubin", value=0.5)
alk_phos = st.number_input("Alkaline Phosphotase", value=100)
sgpt = st.number_input("SGPT", value=20)
sgot = st.number_input("SGOT", value=20)
total_protein = st.number_input("Total Protein", value=6.5)
albumin = st.number_input("Albumin", value=3.5)
ag_ratio = st.number_input("Albumin/Globulin Ratio", value=1.0)

if st.button("Predict Liver Disease"):
    gender_encoded = LabelEncoder().fit(["Male", "Female"]).transform([gender])[0]
    liver_input = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                             alk_phos, sgot, sgpt, total_protein, albumin, ag_ratio]])
    liver_input_scaled = liver_scaler.transform(liver_input)
    liver_pred = liver_model.predict(liver_input_scaled)
    liver_prob = liver_model.predict_proba(liver_input_scaled)
    st.success(f"Prediction: {'Liver Disease' if liver_pred[0]==1 else 'No Liver Disease'}")
    st.info(f"Probability: {liver_prob[0][1]:.2f}")

# ------------------ Kidney Disease Prediction ------------------
st.header("Kidney Disease Prediction")

# Numeric Inputs
age_k = st.number_input("Age", value=30, key='age_k')
bp = st.number_input("Blood Pressure", value=80)
sg = st.number_input("Specific Gravity", value=1.02)
al = st.number_input("Albumin", value=1)
su = st.number_input("Sugar", value=0)
bgr = st.number_input("Blood Glucose Random", value=120)
bu = st.number_input("Blood Urea", value=40)
sc = st.number_input("Serum Creatinine", value=1.2)
sod = st.number_input("Sodium", value=135)
pot = st.number_input("Potassium", value=4.5)
hemo = st.number_input("Hemoglobin", value=15)
pcv = st.number_input("PCV", value=45)
wc = st.number_input("WBC count", value=8000)
rc = st.number_input("RBC count", value=5)

# Categorical Inputs
rbc = st.selectbox("RBC", ['normal', 'abnormal'])
pc = st.selectbox("PC", ['normal', 'abnormal'])
pcc = st.selectbox("PCC", ['present', 'notpresent'])
ba = st.selectbox("BA", ['present', 'notpresent'])
htn = st.selectbox("HTN", ['yes', 'no'])
dm = st.selectbox("DM", ['yes', 'no'])
cad = st.selectbox("CAD", ['yes', 'no'])
appet = st.selectbox("Appetite", ['good', 'poor'])
pe = st.selectbox("PE", ['yes', 'no'])
ane = st.selectbox("ANE", ['yes', 'no'])

def encode_feature(value, mapping_list):
    le = LabelEncoder()
    le.fit(mapping_list)
    return le.transform([value])[0]

if st.button("Predict Kidney Disease"):
    # Encode categorical features
    rbc_encoded = encode_feature(rbc, ['normal','abnormal'])
    pc_encoded = encode_feature(pc, ['normal','abnormal'])
    pcc_encoded = encode_feature(pcc, ['notpresent','present'])
    ba_encoded = encode_feature(ba, ['notpresent','present'])
    htn_encoded = encode_feature(htn, ['no','yes'])
    dm_encoded = encode_feature(dm, ['no','yes'])
    cad_encoded = encode_feature(cad, ['no','yes'])
    appet_encoded = encode_feature(appet, ['good','poor'])
    pe_encoded = encode_feature(pe, ['no','yes'])
    ane_encoded = encode_feature(ane, ['no','yes'])

    # Arrange features in exact order as trained model (after dropping 'id' and target)
    # Order: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
    kidney_input_np = np.array([[age_k, bp, sg, al, su,
                                 rbc_encoded, pc_encoded, pcc_encoded, ba_encoded,
                                 bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                 htn_encoded, dm_encoded, cad_encoded, appet_encoded,
                                 pe_encoded, ane_encoded]])
    
    kidney_input_scaled = kidney_scaler.transform(kidney_input_np)
    kidney_pred = kidney_model.predict(kidney_input_scaled)
    kidney_prob = kidney_model.predict_proba(kidney_input_scaled)
    st.success(f"Prediction: {'Kidney Disease' if kidney_pred[0]==1 else 'No Kidney Disease'}")
    st.info(f"Probability: {kidney_prob[0][1]:.2f}")

# ------------------ Parkinson's Disease Prediction ------------------
st.header("Parkinson's Disease Prediction")

parkinsons_features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
                       'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3',
                       'Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2',
                       'D2','PPE']

parkinsons_input = []
for feature in parkinsons_features:
    val = st.number_input(feature, value=0.0, key=feature)
    parkinsons_input.append(val)

if st.button("Predict Parkinson's Disease"):
    parkinsons_input_np = np.array([parkinsons_input])
    parkinsons_input_scaled = parkinsons_scaler.transform(parkinsons_input_np)
    parkinsons_pred = parkinsons_model.predict(parkinsons_input_scaled)
    parkinsons_prob = parkinsons_model.predict_proba(parkinsons_input_scaled)
    st.success(f"Prediction: {'Parkinsons Disease' if parkinsons_pred[0]==1 else 'No Parkinsons Disease'}")
    st.info(f"Probability: {parkinsons_prob[0][1]:.2f}")