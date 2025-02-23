import unittest
from unittest.mock import MagicMock, patch
import src.alpr
import cv2
import numpy as np
import configparser
import requests
from io import BytesIO
from PIL import Image

# Create a dummy config file for testing.
test_config = configparser.ConfigParser()
test_config['API'] = {'APIKey': 'dummy_key', 'APIEndpoint': 'https://example.com/api'}  # Use a mock endpoint
test_config['Database'] = {} #Not used.
test_config['Logging'] = {} #Not used.
test_config['Camera'] = {} #Not used.

class TestALPRProcessor(unittest.TestCase):
    @patch('src.alpr.requests.post')
    def test_process_frame_api_success(self, mock_post):
        """Test successful API-based license plate detection and OCR."""

        mock_db_conn = MagicMock()
        alpr_processor = src.alpr.ALPRProcessor(mock_db_conn, test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "TEST1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mock the API response and pytesseract
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'candidates': [
                {
                    'content':{
                        'parts':[
                            {'text': '```json\n{\n  "plate": "TEST1234"\n}\n```'}
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with patch('src.alpr.pytesseract.image_to_string', return_value="TEST1234"):
            result = alpr_processor.process_frame(dummy_image)

        self.assertIsNotNone(result)
        self.assertEqual(result['plate_number'], "TEST1234")
        mock_db_conn.insert_plate_data.assert_called() # Check that data insertion attempt happened


    @patch('src.alpr.requests.post')
    def test_process_frame_api_failure(self, mock_post):
        """Test cases where the API call fails."""

        mock_db_conn = MagicMock()
        alpr_processor = src.alpr.ALPRProcessor(mock_db_conn, test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)

        # 1. Test HTTP error
        mock_post.side_effect = requests.exceptions.RequestException("Mocked HTTP error")
        result = alpr_processor.process_frame(dummy_image)
        self.assertIsNone(result)
        mock_post.side_effect = None

        # 2. Test invalid JSON response
        mock_response = MagicMock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        result = alpr_processor.process_frame(dummy_image)
        self.assertIsNone(result)


    @patch('src.alpr.requests.post')
    def test_perform_ocr_success(self, mock_post):
        mock_db_conn = MagicMock()
        alpr_processor = src.alpr.ALPRProcessor(mock_db_conn, test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "TEST1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #Since the detect_license_plate_api returns pillow image. No need to convert
        image = Image.fromarray(dummy_image)
        with patch('src.alpr.pytesseract.image_to_string', return_value="TEST1234") as mock_ocr:
            result = alpr_processor.perform_ocr(image)
            mock_ocr.assert_called()
            self.assertEqual(result, "TEST1234")


    @patch('src.alpr.requests.post')
    def test_perform_ocr_failure(self, mock_post):

        mock_db_conn = MagicMock()
        alpr_processor = src.alpr.ALPRProcessor(mock_db_conn, test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_image)

        with patch('src.alpr.pytesseract.image_to_string', side_effect=Exception("OCR failed")):
            result = alpr_processor.perform_ocr(image)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()