import base64
import numpy as np
import cv2
import requests
import time
import google.auth
import google.auth.transport.requests
from google import genai
from google.genai import types

class Models:
    def __init__(self, project_id, gemini_location="global", imagen_location="us-central1"):
        self.project_id = project_id
        self.gemini_location = gemini_location
        self.imagen_location = imagen_location
        self.gemini_client = genai.Client(vertexai=True, project=project_id, location=gemini_location)

    def upscale_gemini(self, image_bytes, tier="1K", retries=3):
        for attempt in range(retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-3.1-flash-image-preview",
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        "You are an expert image upscaler. Please upscale this input image to high resolution."
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                        image_config=types.ImageConfig(image_size=tier)
                    )
                )
                for i, part in enumerate(response.candidates[0].content.parts):
                    if part.inline_data:
                        return cv2.imdecode(np.frombuffer(part.inline_data.data, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        print(f"  [GEMINI ERROR] Attempt {attempt+1}/{retries}: Part {i} has no inline_data.")
            except Exception as e:
                print(f"  [GEMINI EXCEPTION] Attempt {attempt+1}/{retries}: ({type(e).__name__}): {e}")
                time.sleep(2)
        return None

    def upscale_imagen(self, image_bytes, scale_factor="x4", retries=3):
        """Upscales using Imagen 4.0 Upscale with retry logic and 17MP limit safety."""
        # 17MP limit safety: width * height * factor^2 <= 17,000,000
        # For x4, input must be <= ~1.06MP
        scale = int(scale_factor.replace('x', ''))
        max_input_pixels = 17_000_000 / (scale * scale)
        
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w * h > max_input_pixels:
            ratio = np.sqrt(max_input_pixels / (w * h))
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
            print(f"    [SAFETY] Resized Imagen input to {new_w}x{new_h} to stay under 17MP limit.")
        else:
            # Standardize input as JPEG to reduce payload size and potentially improve API stability
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()

        for attempt in range(retries):
            try:
                creds, _ = google.auth.default()
                auth_req = google.auth.transport.requests.Request()
                creds.refresh(auth_req)
                
                # Use regional endpoint as per documentation for better reliability
                url = f"https://{self.imagen_location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.imagen_location}/publishers/google/models/imagen-4.0-upscale-preview:predict"
                headers = {
                    "Authorization": f"Bearer {creds.token}",
                    "Content-Type": "application/json; charset=utf-8"
                }
                
                payload = {
                    "instances": [{"prompt": "Upscale the image", "image": {"bytesBase64Encoded": base64.b64encode(image_bytes).decode('utf-8')}}],
                    "parameters": {"mode": "upscale", "upscaleConfig": {"upscaleFactor": scale_factor}}
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    if "predictions" in data and len(data["predictions"]) > 0:
                        pred = data["predictions"][0]
                        if "bytesBase64Encoded" in pred and pred["bytesBase64Encoded"]:
                            img_b64 = pred["bytesBase64Encoded"]
                            return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
                        else:
                            print(f"  [IMAGEN ERROR] Prediction object is missing bytes. Full Object: {pred}")
                    else:
                        print(f"  [IMAGEN ERROR] Predictions list empty. Full Response JSON: {data}")
                elif response.status_code == 429:
                    print(f"  [IMAGEN RATE LIMITED] Retry {attempt+1}/{retries} in 5s...")
                    time.sleep(5)
                else:
                    print(f"  [IMAGEN API ERROR] Status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"  [IMAGEN TIMEOUT] Retry {attempt+1}/{retries}...")
                time.sleep(2)
            except Exception as e:
                print(f"  [IMAGEN EXCEPTION] ({type(e).__name__}): {e}")
                time.sleep(1)
                
        return None

    def run_tier(self, model_name, image_bytes, tier="1K"):
        if model_name.lower() == "nano banana 2":
            return self.upscale_gemini(image_bytes, tier=tier)
        return self.upscale_imagen(image_bytes, scale_factor="x4")
