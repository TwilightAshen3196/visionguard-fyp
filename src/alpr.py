import cv2
import pytesseract
import requests
import json
import src.utils as utils
import base64
from io import BytesIO
from PIL import Image


class ALPRProcessor:
    def __init__(self, db_conn, config):
        self.db_conn = db_conn
        self.config = config
        self.api_key = config['API']['APIKey']
        self.api_endpoint = config['API']['APIEndpoint']
        utils.log_message(f"Using API for license plate detection at {self.api_endpoint}.")


    def process_frame(self, frame):
        license_plate_image = self.detect_license_plate_api(frame)

        if license_plate_image is None:
            return None

        plate_number = self.perform_ocr(license_plate_image)
        if not plate_number:
            return None

        timestamp = utils.get_current_timestamp()
        plate_data = {
            'plate_number': plate_number,
            'image_path': 'N/A',
            'detection_time': timestamp,
            'location': 'N/A',
            'user_id': 'N/A'
        }
        db_id = self.db_conn.insert_plate_data(plate_data)
        plate_data['id'] = db_id
        return plate_data

    def detect_license_plate_api(self, frame):
        try:
            _, encoded_image = cv2.imencode(".jpg", frame)
            image_bytes = encoded_image.tobytes()
            base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Extract the license plate number from this image. Respond only with the license plate number, nothing else."}, # Simpler, more direct prompt
                            {"inline_data": {"mime_type": "image/jpeg", "data": base64_encoded}},
                        ]
                    }
                ]
            }

            response = requests.post(self.api_endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()
            utils.log_message(f"Gemini API response: {response_json}")

            # --- Robust Response Parsing (VERY IMPORTANT) ---
            plate_text = self.extract_plate_text(response_json)
            if plate_text:
                 image = Image.open(BytesIO(image_bytes))
                 return image
            else:
                return None


        except requests.exceptions.RequestException as e:
            utils.log_message(f"Error calling Gemini API: {e}", level="ERROR")
            return None
        except Exception as e:
            utils.log_message(f"An unexpected error occurred during API detection: {e}", "ERROR")
            return None

    def extract_plate_text(self, response_json):
        """
        Robustly extracts the license plate text from the Gemini API response.
        Handles various potential response structures and error conditions.
        """
        try:
            # Check for errors in the response first
            if "error" in response_json:
                error_message = response_json["error"].get("message", "Unknown error")
                utils.log_message(f"Gemini API returned an error: {error_message}", level="ERROR")
                return None

            # Check for candidates and content
            if "candidates" not in response_json or not response_json["candidates"]:
                utils.log_message("Gemini API response missing 'candidates' or 'candidates' is empty.", level="WARNING")
                return None

            candidate = response_json["candidates"][0]  # Get the first candidate

            if "content" not in candidate or "parts" not in candidate["content"]:
                utils.log_message("Gemini API response missing 'content' or 'parts' in candidate.", level="WARNING")
                return None

            parts = candidate["content"]["parts"]
            if not parts:
                utils.log_message("Gemini API: 'parts' is empty in candidate.", level="WARNING")
                return None


            # Iterate through parts to find text
            for part in parts:
                if "text" in part:
                    plate_text = part["text"].strip()
                    # Basic validation: check for empty string and potentially filter out very short strings
                    if plate_text and len(plate_text) > 2:  # Consider plates with at least 3 characters
                        return plate_text

            utils.log_message("Gemini API: No text found in any 'parts' of the response.", level="WARNING")
            return None

        except (KeyError, IndexError, TypeError) as e:
            utils.log_message(f"Error parsing Gemini API response: {e}", level="ERROR")
            utils.log_message(f"Problematic response JSON: {response_json}", level="DEBUG") # Log the full response for debugging
            return None


    def perform_ocr(self, image):
        try:
            gray_image = image.convert('L')
            text = pytesseract.image_to_string(gray_image, config='--psm 6')
            text = text.strip()
            utils.log_message(f"OCR result: {text}")
            return text
        except Exception as e:
            utils.log_message(f"Error during OCR: {e}", level="ERROR")
            return None