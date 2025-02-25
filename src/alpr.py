import cv2
import easyocr
import src.utils as utils
from ultralytics import YOLO  # Import YOLO
import numpy as np

class ALPRProcessor:
    def __init__(self, config): # No db_conn passed here anymore.
        self.config = config
        self.model = YOLO('yolov8n.pt')  # Load a pre-trained YOLOv8n model
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English
        utils.log_message("Using local YOLOv8 and EasyOCR for license plate detection and recognition.")


    def process_frame(self, frame, db_conn): # db_conn is now passed as argument
        # 1. License Plate Detection (YOLOv8)
        results = self.model(frame)  # Run YOLOv8 inference
        license_plate_coords = self.extract_license_plate_coordinates(results)

        if license_plate_coords is None:
            return None  # No plate detected

        # 2. Crop and OCR (EasyOCR)
        for x1, y1, x2, y2 in license_plate_coords:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            license_plate_crop = frame[y1:y2, x1:x2]

            # Run EasyOCR on the cropped image
            ocr_result = self.reader.readtext(license_plate_crop)

            plate_number = self.extract_plate_number(ocr_result) # Extract the plate number
            if not plate_number:
                continue # If no plate was found, continue to the next detection.


            # 3. Data Logging (Database) - Use the passed db_conn
            timestamp = utils.get_current_timestamp()
            plate_data = {
                'plate_number': plate_number,
                'image_path': 'N/A',
                'detection_time': timestamp,
                'location': 'N/A',
                'user_id': 'N/A'
            }
            db_id = db_conn.insert_plate_data(plate_data) # Use the passed db_conn to insert
            plate_data['id'] = db_id
            return plate_data # Return the first detected plate

        return None # Nothing detected

    def extract_license_plate_coordinates(self, results):
        """Extracts bounding box coordinates of detected license plates."""
        coordinates = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Filter the current object class, if it is not a car, then continue
                if result.names[int(box.cls[0])] != 'car':
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                coordinates.append((x1, y1, x2, y2))
        return coordinates
    def extract_plate_number(self, ocr_result):
        """
        Extracts the license plate number from the EasyOCR result.
        Applies filtering and returns the most likely candidate.
        """
        if not ocr_result:
            return None

        best_candidate = ""
        highest_confidence = 0.0

        for (bbox, text, prob) in ocr_result:
            # Basic filtering: Remove spaces and common OCR errors
            text = text.replace(" ", "").upper()
            text = text.replace("O", "0")  # Common OCR mistake: O instead of 0
            text = text.replace("I", "1")  # Common OCR mistake: I instead of 1
            text = text.replace(" ", "")   #Removes spaces

            # Check for alphanumeric characters (you can customize this)
            if text.isalnum() and 3 <= len(text) <= 10:  # Example length constraint
                if prob > highest_confidence:
                    best_candidate = text
                    highest_confidence = prob
        utils.log_message(f"Extracted text: {best_candidate}, confidence = {highest_confidence}")
        return best_candidate