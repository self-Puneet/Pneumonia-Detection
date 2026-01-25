import argparse
import requests
import sys
import json


def upload_image(url: str, image_path: str, field_name: str = "image"):
    with open(image_path, "rb") as f:
        files = {field_name: (image_path.split("/")[-1], f, "image/jpeg")}
        resp = requests.post(url, files=files)
    return resp


def main():
    parser = argparse.ArgumentParser(description="Upload image to Pneumonia Detection API")
    parser.add_argument("--url", default="http://localhost:5000/predict", help="Full URL to the /predict endpoint")
    parser.add_argument("--image", default="test_person78_bacteria_382.jpeg", help="Path to the image to upload")
    args = parser.parse_args()

    print("="*70)
    print("🧪 PNEUMONIA DETECTION API TEST")
    print("="*70)
    print(f"📡 URL: {args.url}")
    print(f"📸 Image: {args.image}")
    print()

    try:
        print("⏳ Uploading image and waiting for prediction...")
        resp = upload_image(args.url, args.image)
    except FileNotFoundError:
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(2)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Request failed: {e}")
        sys.exit(3)

    print()
    print("="*70)
    print(f"📊 RESPONSE (HTTP {resp.status_code})")
    print("="*70)
    
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        
        # Extra readable summary if successful
        if resp.status_code == 200 and data.get('success'):
            print()
            print("="*70)
            print("📋 SUMMARY")
            print("="*70)
            has_pneumonia = data.get('has_pneumonia', False)
            confidence = data.get('confidence', 0)
            
            result_emoji = "⚠️" if has_pneumonia else "✅"
            result_text = "PNEUMONIA DETECTED" if has_pneumonia else "NORMAL (No Pneumonia)"
            
            print(f"{result_emoji} Result: {result_text}")
            print(f"📈 Confidence: {confidence:.2f}%")
            
            probs = data.get('probabilities', {})
            print(f"\nProbabilities:")
            print(f"  • Normal:    {probs.get('normal', 0):.2f}%")
            print(f"  • Pneumonia: {probs.get('pneumonia', 0):.2f}%")
            
            print(f"\n⏱️ Processing Time: {data.get('processing_time', 'N/A')}")
            print("="*70)
        
    except ValueError:
        print("❌ Invalid JSON response:")
        print(resp.text)


if __name__ == "__main__":
    main()
