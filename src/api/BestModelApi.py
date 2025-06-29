from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Create the application
app = FastAPI(title="Student Performance Prediction API", 
              description="Predict student grade class based on various factors")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from any source. Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (HTML files)
templates = Jinja2Templates(directory="templates")

# Load the trained enhanced model
model_data = joblib.load("models/enhanced_student_perf_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main index.html page (the website)"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict(
    Age: int = Query(..., description="Student age (15 to 18 years)"),
    Gender: int = Query(..., description="Student gender (0: Male, 1: Female)"),
    Ethnicity: int = Query(..., description="Ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)"),
    ParentalEducation: int = Query(..., description="Parental education level (0: None, 1: High School, 2: Some College, 3: Bachelor's, 4: Higher)"),
    StudyTimeWeekly: float = Query(..., description="Weekly study time (in hours, 0 to 20)"),
    Absences: int = Query(..., description="Number of absences during academic year (0 to 30)"),
    Tutoring: int = Query(..., description="Receiving tutoring (0: No, 1: Yes)"),
    ParentalSupport: int = Query(..., description="Parental support level (0: None, 1: Low, 2: Moderate, 3: High, 4: Very High)"),
    Extracurricular: int = Query(..., description="Participation in extracurricular activities (0: No, 1: Yes)"),
    Sports: int = Query(..., description="Participation in sports activities (0: No, 1: Yes)"),
    Music: int = Query(..., description="Participation in music activities (0: No, 1: Yes)"),
    Volunteering: int = Query(..., description="Participation in volunteering (0: No, 1: Yes)"),
):
    # Feature engineering function (same as in training)
    def create_additional_features(df):
        df_enhanced = df.copy()
        
        # Interaction features
        if 'StudyTimeWeekly' in df.columns and 'Absences' in df.columns:
            df_enhanced['StudyTime_per_Absence'] = df_enhanced['StudyTimeWeekly'] / (df_enhanced['Absences'] + 1)
        
        # Age grouping
        if 'Age' in df.columns:
            df_enhanced['Age_Group'] = pd.cut(df_enhanced['Age'], bins=3, labels=['Young', 'Medium', 'Old'])
            df_enhanced['Age_Group'] = LabelEncoder().fit_transform(df_enhanced['Age_Group'])
        
        return df_enhanced
    
    # Collect data in dictionary
    data = {
        "Age": Age,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "ParentalEducation": ParentalEducation,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "Tutoring": Tutoring,
        "ParentalSupport": ParentalSupport,
        "Extracurricular": Extracurricular,
        "Sports": Sports,
        "Music": Music,
        "Volunteering": Volunteering,
    }
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Apply feature engineering
    df_enhanced = create_additional_features(df)
    
    # Ensure columns are in the same order as training
    feature_columns = model_data['feature_names']
    df_enhanced = df_enhanced[feature_columns]
    
    # Scale the data if required
    if model_data['requires_scaling']:
        df_scaled = model_data['scaler'].transform(df_enhanced)
    else:
        df_scaled = df_enhanced.values
    
    # Make prediction
    model = model_data['model']
    prediction = model.predict(df_scaled)[0]
    prediction_proba = model.predict_proba(df_scaled)[0]
    
    # Grade mapping
    grade_map = {
        0: 'Excellent (90-100%)',
        1: 'Very Good (80-89%)',
        2: 'Good (70-79%)',
        3: 'Acceptable (60-69%)',
        4: 'Fail (Below 60%)'
    }
    
    grade = grade_map.get(prediction, 'Unknown')
    confidence = float(max(prediction_proba))
    
    return {
        "predicted_grade": grade,
        "confidence": f"{confidence:.2%}",
        "prediction_probabilities": {
            grade_map[i]: f"{prob:.2%}" for i, prob in enumerate(prediction_proba)
        },
        "model_used": model_data['model_name']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
