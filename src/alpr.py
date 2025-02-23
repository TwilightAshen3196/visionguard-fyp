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
                            {"text": "Extract the license plate number as JSON {\\\"plate\\\": \\\"result\\\"} from this image."},
                            {"inline_data": {"mime_type": "image/jpeg", "data": base64_encoded}},
                        ]
                    }
                ]
            }

            response = requests.post(self.api_endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            response_json = response.json()
            utils.log_message(f"Gemini API response: {response_json}")

            try:
                text_response = response_json['candidates'][0]['content']['parts'][0]['text']
                start_index = text_response.find('{')
                end_index = text_response.rfind('}') + 1

                if start_index == -1 or end_index == -1:
                     utils.log_message(f"Gemini API response does not conatain the expected JSON format")
                     return None

                json_str = text_response[start_index:end_index]
                result_json = json.loads(json_str)
                plate_text = result_json.get('plate')

                if not plate_text or plate_text.upper() == 'RESULT':
                    return None

                image = Image.open(BytesIO(image_bytes))
                return image

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                utils.log_message(f"Error parsing Gemini API response: {e}", level="ERROR")
                return None

        except requests.exceptions.RequestException as e:
            utils.log_message(f"Error calling Gemini API: {e}", level="ERROR")
            return None
        except Exception as e:
            utils.log_message(f"An unexpected error occurred during API detection: {e}", "ERROR")
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