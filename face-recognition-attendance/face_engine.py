import os
import cv2
import numpy as np

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

class FaceEngine:
    def __init__(self):
        # Verify models exist
        if not os.path.exists(YUNET_MODEL) or not os.path.exists(SFACE_MODEL):
            raise FileNotFoundError(
                "ONNX model files not found. Please run models/download_models.py first."
            )
            
        # 1. Initialize YuNet Face Detector
        # Note: We set a default input size of 320x320. It must be dynamically adjusted 
        # to match the video frame dimensions during processing.
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=(320, 320),
            score_threshold=0.85, # 85% confidence score required
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        # 2. Initialize SFace Face Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_MODEL,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        
        # Current input size tracking
        self.current_input_size = (320, 320)
        
        # SFace Cosine Similarity Threshold:
        # Cosine score >= 0.363 means it's a match. Higher is more strict/confident.
        # We will use 0.37 for a slightly more strict/confident match locally.
        self.match_threshold = 0.37

    def set_frame_size(self, width, height):
        """Update YuNet detector input size dynamically to match frame dimensions"""
        if self.current_input_size != (width, height):
            self.detector.setInputSize((width, height))
            self.current_input_size = (width, height)

    def detect_faces(self, frame):
        """
        Detects all faces in the frame.
        Returns a list of dictionaries with bounding boxes and face landmarks.
        """
        h, w = frame.shape[:2]
        self.set_frame_size(w, h)
        
        retval, faces = self.detector.detect(frame)
        
        detected_list = []
        if faces is not None:
            for face in faces:
                # Bounding box coordinates
                x, y, width, height = map(int, face[0:4])
                confidence = float(face[14])
                
                # Check for boundary issues
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)
                
                detected_list.append({
                    "box": (x, y, width, height),
                    "landmarks": face,  # Full landmark array for alignCrop
                    "confidence": confidence
                })
        return detected_list

    def extract_embedding(self, frame, face_landmarks):
        """
        Aligns the face and extracts a 128D numpy array embedding.
        """
        try:
            # Crop and align the face based on the detected landmarks
            aligned_face = self.recognizer.alignCrop(frame, face_landmarks)
            # Extract features (128D float32 array)
            feature = self.recognizer.feature(aligned_face)
            return feature
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None

    def match_face(self, query_feature, enrolled_students):
        """
        Matches a query face embedding against a list of enrolled student records.
        Each student record should be a dict containing 'id', 'name', 'roll_number', and 'face_embedding' (numpy array).
        Returns (match_student_dict, score) if a match is found, or (None, highest_score).
        """
        if not enrolled_students or query_feature is None:
            return None, 0.0
            
        best_match = None
        highest_score = -1.0
        
        for student in enrolled_students:
            enrolled_feature = student["face_embedding"]
            
            # SFace match returns the cosine similarity when using the standard setting
            try:
                # SFace requires features to be reshaped to 1x128 if not already
                f1 = query_feature.reshape(1, -1)
                f2 = enrolled_feature.reshape(1, -1)
                
                score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
                
                if score > highest_score:
                    highest_score = score
                    if score >= self.match_threshold:
                        best_match = student
            except Exception as e:
                print(f"Error matching feature for student {student['name']}: {e}")
                
        return best_match, highest_score

# Test utility
if __name__ == "__main__":
    try:
        engine = FaceEngine()
        print("FaceEngine initialized successfully with YuNet and SFace ONNX models!")
    except Exception as e:
        print(f"Error initializing FaceEngine: {e}")
