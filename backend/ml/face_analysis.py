import os
import random
import numpy as np
import httpx
from backend.ml.base import BaseModelInterface

# Setup deepface conditionally to avoid breaking startup if models are downloading
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class ArcFaceModel(BaseModelInterface):
    """
    Real face analysis using ArcFace via the deepface library.
    Since we only have a single image, this model will:
    1. Detect if a face exists. No face = high risk (for a profile photo).
    2. Detect if multiple faces exist. Multiple faces = medium risk.
    3. If one face is found, we extract the ArcFace embedding and generate
       a simulated similarity score against a mocked "known bot database".
    """
    def __init__(self):
        self.model_name = "ArcFace"
        self.detector_backend = "opencv"

    def predict(self, features: dict) -> float:
        """
        Returns a risk score between 0.0 (legitimate) and 1.0 (high risk).
        """
        profile_image_url = features.get("profile_image_url", "")
        if not profile_image_url:
            return 0.8  # No profile image provided = high risk
            
        if not DEEPFACE_AVAILABLE:
            # Fallback to dummy if deepface isn't installed/working
            return self._dummy_predict(features)

        # Download the image temporarily
        img_path = self._download_image(profile_image_url)
        if not img_path:
            return 0.9  # Invalid/broken image URL

        try:
            # 1. Face Detection and Embedding Extraction
            # enforce_detection=True will throw an exception if no face is found
            face_objs = DeepFace.represent(
                img_path=img_path, 
                model_name=self.model_name, 
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # 2. Analyze the faces
            if not face_objs or len(face_objs) == 0:
                # No face detected in profile photo
                risk_score = 0.85 
            elif len(face_objs) > 1:
                # Multiple faces in profile photo
                risk_score = 0.60
            else:
                # Exactly one face detected
                # Extract the embedding vector (e.g. 512 dimensions for ArcFace)
                embedding = face_objs[0].get("embedding", [])
                
                # Mock similarity to known fake faces (Option B)
                # In production, this would be: cosine_similarity(embedding, known_bot_db)
                # Here we use a deterministic hash of the embedding's first few values to mock a score
                if embedding:
                    mock_hash = sum(embedding[:10]) * 1000
                    # Pseudo-random but deterministic risk between 0.05 and 0.95
                    risk_score = 0.05 + (abs(mock_hash) % 90) / 100.0
                else:
                    risk_score = 0.5

        except Exception as e:
            # Error during processing (e.g., corrupted image)
            print(f"DeepFace processing error: {e}")
            risk_score = 0.7
        finally:
            # Cleanup temp image
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except:
                    pass

        return round(risk_score, 2)

    def _download_image(self, url: str) -> str:
        """Downloads image to a temp file and returns the path."""
        # For mock profile URLs (like http://example.com/bot.jpg), return None to simulate failure 
        # unless it's a real URL or we are testing
        if "example.com" in url or not url.startswith("http"):
            # We don't actually download mock URLs. We'll generate a dummy temp file instead
            # just so deepface doesn't crash on file not found.
            return self._create_dummy_image(url)

        try:
            # Real download
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                temp_name = f"temp_{hash(url)}.jpg"
                with open(temp_name, "wb") as f:
                    f.write(response.content)
                return temp_name
        except Exception:
            pass
        return ""
        
    def _create_dummy_image(self, url: str) -> str:
        """Create a solid color image for testing purposes."""
        import cv2
        temp_name = f"temp_mock_{hash(url)}.jpg"
        
        # If the URL implies a bot, maybe we don't draw a face, 
        # but for now we just create a blank image which DeepFace will detect as "no face"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (200, 200, 200) # Gray background
        
        cv2.imwrite(temp_name, img)
        return temp_name

    def _dummy_predict(self, features: dict) -> float:
        username = features.get("username", "")
        if "bot" in username.lower():
            return round(random.uniform(0.7, 0.95), 2)
        return round(random.uniform(0.05, 0.3), 2)
