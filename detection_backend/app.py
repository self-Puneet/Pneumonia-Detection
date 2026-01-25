from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from services.predict_service import predict_image, load_models

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models once at startup
print("=" * 70)
print("🚀 LOADING MODELS AT STARTUP")
print("=" * 70)
load_models()
print("=" * 70)
print("✅ ALL MODELS LOADED - READY TO SERVE")
print("=" * 70)

@app.route("/api", methods=["GET"])
def home():
    return jsonify({
        "message": "Pneumonia Detection API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Upload chest X-ray image for pneumonia detection"
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Check if image is present
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400

        # Call prediction service
        result = predict_image(image_file)

        return jsonify({
            "success": True,
            "has_pneumonia": result["has_pneumonia"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "features": result["features"],
            "processing_time": result["processing_time"]
        }), 200

    except Exception as e:
        # Log the error
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
