# Multiple Disease Prediction System

A machine learning-based web application for predicting Liver Disease, Kidney Disease, and Parkinson's Disease using Streamlit. The system uses pre-trained Random Forest models to provide predictions based on user-inputted medical features.

## Features

- **Liver Disease Prediction**: Predicts the likelihood of liver disease based on features like age, gender, bilirubin levels, and enzyme values.
- **Kidney Disease Prediction**: Assesses kidney health using inputs such as blood pressure, specific gravity, albumin, and categorical factors like hypertension and diabetes.
- **Parkinson's Disease Prediction**: Evaluates Parkinson's risk from voice-related features (e.g., jitter, shimmer, and other acoustic measures).
- **User-Friendly Interface**: Built with Streamlit for easy input and real-time predictions.
- **Model Accuracy**: Displays prediction probabilities for transparency.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multiple-disease-prediction.git
   cd multiple-disease-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   If `requirements.txt` doesn't exist, install manually:
   ```
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. Download datasets (if not included):
   - Place `indian_liver_patient.csv`, `kidney_disease.csv`, and `parkinsons.csv` in the `data/` folder.
   - Sources: UCI Machine Learning Repository (Liver, Kidney, Parkinson's datasets).

4. Train models (optional, if models aren't pre-trained):
   ```
   python scripts/preprocess_train.py
   ```
   This generates model files in `models/`.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your browser (usually `http://localhost:8501`).

3. Navigate to the desired disease prediction section, enter the required features, and click "Predict".

### Input Examples
- **Liver Disease**: Age (e.g., 30), Gender (Male/Female), Total Bilirubin (1.0), etc.
- **Kidney Disease**: Age (30), Blood Pressure (80), Specific Gravity (1.02), etc., plus categorical inputs like RBC (normal/abnormal).
- **Parkinson's Disease**: Voice features like MDVP:Fo(Hz) (197.076), MDVP:Fhi(Hz) (206.896), etc.

## Training Models

Run `scripts/preprocess_train.py` to preprocess data, train models, and save them:
- Preprocessing includes handling missing values, encoding categoricals, and scaling.
- Models are saved as `.pkl` files in `models/` (e.g., `liver_model.pkl`).
- Scalers are also saved for consistent prediction scaling.

## Datasets

- **Liver Disease**: `indian_liver_patient.csv` (UCI - Indian Liver Patient Dataset).
- **Kidney Disease**: `kidney_disease.csv` (UCI - Chronic Kidney Disease Dataset).
- **Parkinson's Disease**: `parkinsons.csv` (UCI - Parkinson's Dataset).

Ensure datasets are in the `data/` folder. The app expects these exact filenames.

## Technologies Used

- **Python**: Core language.
- **Streamlit**: For the web interface.
- **Scikit-Learn**: For model training (Random Forest) and preprocessing.
- **Pandas & NumPy**: For data handling.
- **Joblib**: For model serialization.

## Project Structure

```
multiple-disease-prediction/
├── app.py                          # Main Streamlit app
├── scripts/
│   └── preprocess_train.py         # Data preprocessing and model training
├── models/                         # Trained models and scalers (generated)
├── data/                           # Datasets (not included in repo)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

