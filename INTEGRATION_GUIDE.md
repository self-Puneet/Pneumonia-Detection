# Backend and Frontend Integration - Setup Guide

## ✅ Completed Integration

The pneumonia detection system is now fully integrated between the backend (Flask) and frontend (Flutter). Here's what was implemented:

### Backend Changes:

1. **requirements.txt** - Added all necessary dependencies:
   - Flask & Flask-CORS for API
   - OpenCV, NumPy for image processing
   - scikit-learn for ML model
   - MediaPipe for lung segmentation

2. **services/predict_service.py** - Complete prediction pipeline:
   - Image preprocessing with CLAHE
   - Lung segmentation using MediaPipe
   - 43 edge-based feature extraction
   - Random Forest model prediction
   - Returns: pneumonia status, confidence, probabilities, features, processing time

3. **app.py** - Updated Flask API:
   - Added CORS support for web requests
   - Proper error handling
   - Returns structured JSON response with:
     - `has_pneumonia`: boolean
     - `confidence`: percentage
     - `probabilities`: normal & pneumonia percentages  
     - `features`: all 43 extracted features
     - `processing_time`: time taken

### Frontend Changes:

1. **pubspec.yaml** - Added http package for API calls

2. **lib/services/api_service.dart** - New API service:
   - Handles image upload to backend
   - Processes prediction response
   - Connection checking utility

3. **lib/pages/home_page.dart** - Updated to use real API:
   - Replaced mock data with actual API calls
   - Real-time prediction from backend
   - Error handling for connection issues
   - Download features JSON with real data

## 🚀 How to Run

### Step 1: Start the Backend

```bash
# Navigate to backend directory
cd "d:\College\sem 5\Pneumonia-Detection\detection_backend"

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the Flask server
python app.py
```

The backend will start on `http://localhost:5000`

### Step 2: Start the Frontend

```bash
# Navigate to Flutter app directory  
cd "d:\College\sem 5\Pneumonia-Detection\pneumonia_detection_app"

# Get packages (first time only)
flutter pub get

# Run the app
flutter run -d chrome --web-port=8080
```

The frontend will open in Chrome at `http://localhost:8080`

## 📡 API Endpoints

### GET /
- Health check endpoint
- Returns API information

### POST /predict
- Upload chest X-ray image
- Form data: `image` (file)
- Returns:
```json
{
  "success": true,
  "has_pneumonia": true,
  "confidence": 87.42,
  "probabilities": {
    "normal": 12.58,
    "pneumonia": 87.42
  },
  "features": {
    "RL_edge_completeness": 0.85,
    "RL_edge_smoothness": 0.72,
    ...
  },
  "processing_time": "1.85s"
}
```

## 🧪 Testing the Integration

1. **Start both servers** (backend and frontend)
2. **Upload an X-ray image** in the Flutter app
3. **Click Predict** - The app will call the backend API
4. **View results**:
   - Pneumonia detection (Yes/No)
   - Confidence percentage
   - Processing time
   - All 43 extracted features
5. **Download features** - Click the download button to get a JSON file with all features

## 🔧 Troubleshooting

### Backend Issues:

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"Model file not found":**
- Make sure `random_forest_model.pkl` exists in `detection_backend/` folder
- Or `decision_tree_model.pkl` in `models_output/` folder

**"Segmentation failed":**
- Ensure `segmentation.py` and `weights.pt` are in the parent directory

### Frontend Issues:

**"Connection refused":**
- Make sure backend is running on port 5000
- Check the `baseUrl` in `lib/services/api_service.dart`

**CORS errors:**
- Backend has CORS enabled via Flask-CORS
- If issues persist, check browser console

## 🎯 Features Included

### Backend:
- ✅ 43 edge-based feature extraction
- ✅ CLAHE preprocessing
- ✅ Lung segmentation with MediaPipe
- ✅ Random Forest classification
- ✅ Comprehensive error handling
- ✅ Processing time tracking

### Frontend:
- ✅ File upload with validation
- ✅ Real-time prediction display
- ✅ Confidence visualization
- ✅ Feature list with top features
- ✅ Download features as JSON
- ✅ Loading states and error handling
- ✅ Responsive design

## 📝 Notes

- The backend uses the trained Random Forest model
- All 43 features are extracted from the uploaded X-ray
- Features can be downloaded as JSON for analysis
- Processing time typically 1-3 seconds
- Supports JPG, PNG image formats
