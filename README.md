# Student Performance Predictor

A modern web application that predicts student academic performance using advanced machine learning techniques.

## 🚀 Quick Start

**Want to see the website (index.html) immediately?**

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start the server**: `python main.py`
3. **Open your browser**: Go to `http://localhost:8000` 
4. **You'll see the index.html page**: The beautiful Student Performance Predictor website!
5. **Try the predictor**: Fill in the form and click "Predict Performance"!

✅ **The FastAPI server automatically serves:**
- `index.html` at the root URL (`http://localhost:8000`)
- Static files (CSS, JS) from `/static/` directory
- API endpoints like `/predict` for the form submissions

That's it! The website should be running and ready to use.

---

## 🌟 Features

- **Modern Web Interface**: Beautiful, responsive UI with smooth animations
- **Advanced ML Model**: Uses ensemble methods with 97% accuracy
- **Real-time Predictions**: Instant grade predictions based on student data
- **Interactive Elements**: Sliders, dropdowns, and checkboxes for easy data input
- **Professional Design**: Clean, modern interface with help modals and animations

## 📁 Project Structure

```
student_performance_api/
├── data/                           # Dataset files
│   └── Student_performance_data _.csv
├── models/                         # Trained ML models
│   ├── enhanced_student_perf_model.pkl
│   └── student_perf_model.pkl
├── src/                           # Source code
│   ├── api/                       # FastAPI backend
│   │   └── BestModelApi.py
│   └── ml/                        # Machine learning scripts
│       ├── train.py
│       └── train_improved.py
├── static/                        # Frontend assets
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
├── templates/                     # HTML templates
│   └── index.html
├── docs/                          # Documentation
├── logs/                          # Application logs
├── main.py                        # Main entry point
├── train_model.py                 # Training script runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   cd "d:\dev\ML and AI\FastApi\student_performance_api"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using the main entry point (Recommended)
```bash
python main.py
```

#### Option 2: Direct FastAPI execution
```bash
python src/api/BestModelApi.py
```

#### Option 3: Using uvicorn directly
```bash
uvicorn src.api.BestModelApi:app --host 127.0.0.1 --port 8000 --reload
```

### 🌐 Accessing the Website (index.html)

**🎯 IMPORTANT: This is how you access the web application!**

1. **Start the server** using any of the methods above. You'll see output like:
   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   INFO:     Started reloader process [12345] using StatReload
   INFO:     Started server process [67890]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

2. **🌍 Open your web browser** and navigate to the main website:
   ```
   http://localhost:8000
   ```
   or
   ```
   http://127.0.0.1:8000
   ```
   
   **📄 This will load the `index.html` page** - the main web interface!

3. **✨ You should see the Student Performance Predictor website** featuring:
   - **Beautiful Header**: With graduation cap icon and 97% accuracy stats
   - **Student Information Form**: Organized into 4 sections:
     - 👤 **Personal Information**: Age, Gender, Ethnicity
     - 📚 **Academic Information**: Study time (with slider), Absences, Tutoring
     - 🏠 **Family Background**: Parental education and support levels
     - 🏃 **Activities**: Extracurricular, Sports, Music, Volunteering
   - **Interactive Elements**: 
     - Sliders for study time and absences
     - Dropdown menus for selections
     - Beautiful checkboxes for activities
   - **Action Buttons**: 
     - "Predict Performance" (main submit)
     - "Load Sample" (fills form with example data)
     - "Clear Form" (resets all fields)
   - **Help Button**: Floating help button (bottom-right) with detailed instructions

### 🎮 How to Use the Website

4. **📝 Using the Student Performance Predictor**:
   
   **Quick Start:**
   - Click "Load Sample" to see example data
   - Or fill in each section manually:
   
   **Step-by-step:**
   1. **Personal Info**: Select age (15-18), gender, ethnicity
   2. **Academic Info**: 
      - Use the slider for study time (0-20 hours/week)
      - Use the slider for absences (0-30)
      - Select if receiving tutoring (Yes/No)
   3. **Family Background**: Choose parental education and support levels
   4. **Activities**: Check boxes for extracurricular activities
   5. **Predict**: Click "Predict Performance" button
   6. **Results**: View prediction results with confidence scores

   **Website Features:**
   - 🔄 **Real-time sliders**: Drag to adjust study time and absences
   - 📋 **Load Sample**: Fills form with realistic example data
   - 🧹 **Clear Form**: Resets all fields to start fresh
   - ❓ **Help Modal**: Click the floating help button for detailed guidance
   - 📱 **Responsive**: Works on desktop, tablet, and mobile

5. **⏹️ To stop the server**, press `Ctrl+C` in the terminal

## 🔧 Training the Model (Optional)

If you want to retrain the model with new data:

```bash
python train_model.py
```

## 📝 API Usage

### Prediction Endpoint

**GET** `/predict`

**Parameters:**
- `Age`: Student age (15-18)
- `Gender`: 0 (Male) or 1 (Female)
- `Ethnicity`: 0-3 (Caucasian, African American, Asian, Other)
- `ParentalEducation`: 0-4 (None, High School, Some College, Bachelor's, Higher)
- `StudyTimeWeekly`: Hours per week (0-20)
- `Absences`: Number of absences (0-30)
- `Tutoring`: 0 (No) or 1 (Yes)
- `ParentalSupport`: 0-4 (None, Low, Moderate, High, Very High)
- `Extracurricular`: 0 (No) or 1 (Yes)
- `Sports`: 0 (No) or 1 (Yes)
- `Music`: 0 (No) or 1 (Yes)
- `Volunteering`: 0 (No) or 1 (Yes)

**Example Request:**
```
GET http://localhost:8000/predict?Age=17&Gender=1&Ethnicity=2&ParentalEducation=3&StudyTimeWeekly=12.5&Absences=3&Tutoring=1&ParentalSupport=3&Extracurricular=1&Sports=0&Music=1&Volunteering=1
```

**Response:**
```json
{
    "predicted_grade": "Very Good (80-89%)",
    "confidence": "85.4%",
    "prediction_probabilities": {
        "Excellent (90-100%)": "15.2%",
        "Very Good (80-89%)": "85.4%",
        "Good (70-79%)": "12.1%",
        "Acceptable (60-69%)": "5.3%",
        "Fail (Below 60%)": "2.0%"
    },
    "model_used": "Random Forest"
}
```

## 🎯 Web Interface Features

- **Interactive Form**: Fill in student information using intuitive controls
- **Real-time Sliders**: Adjust study time and absences with visual feedback
- **Help Modal**: Click the help button for detailed instructions
- **Sample Data**: Use "Load Sample" button to test with example data
- **Clear Form**: Reset all fields with the "Clear Form" button
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Technologies Used

- **Backend**: FastAPI, Python
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with animations and gradients
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## 📊 Model Information

- **Algorithm**: Ensemble methods (Random Forest, XGBoost, etc.)
- **Accuracy**: 97%
- **Features**: 24 engineered features including interaction terms
- **Data Balancing**: SMOTE for handling class imbalance
- **Cross-validation**: 5-fold stratified validation

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use a different port
   uvicorn src.api.BestModelApi:app --port 8001
   ```

2. **Module not found errors**
   ```bash
   # Make sure you're in the project directory
   cd "d:\dev\ML and AI\FastApi\student_performance_api"
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Model file not found**
   ```bash
   # Retrain the model
   python train_model.py
   ```

## 🚀 Development

To modify the application:

1. **Backend changes**: Edit files in `src/api/`
2. **ML model changes**: Edit files in `src/ml/`
3. **Frontend changes**: Edit files in `static/` and `templates/`
4. **Styling**: Edit `static/css/styles.css`
5. **JavaScript**: Edit `static/js/script.js`

## 📄 License

This project is for educational and demonstration purposes.

## 👨‍💻 Author

Created by Alaa Otay - 2025
