import unittest
from unittest.mock import MagicMock, patch
import src.alpr
import cv2
import numpy as np
import configparser

# Create a dummy config file for testing.
test_config = configparser.ConfigParser()
test_config['Database'] = {}
test_config['Logging'] = {}


class TestALPRProcessor(unittest.TestCase):

    @patch('src.alpr.YOLO') #Mock yolo
    @patch('src.alpr.easyocr.Reader')  # Mock EasyOCR
    @patch('src.alpr.db.connect_to_db') # Mock database connection
    def test_process_frame_success(self, mock_connect_db, mock_easyocr_reader, mock_yolo):
        mock_db_conn = MagicMock() # Mock database connection object.
        mock_connect_db.return_value = mock_db_conn # Make connect_to_db return the mock.

        alpr_processor = src.alpr.ALPRProcessor(test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock YOLO results
        mock_yolo_results = [MagicMock()]
        mock_yolo_results[0].boxes.cpu.return_value.numpy.return_value = [
            MagicMock(cls=[2], xyxy=[[50, 60, 150, 100]])  # Simulate coordinates
        ]
        mock_yolo_results[0].names = {2: 'car'} # car class
        mock_yolo.return_value.return_value = mock_yolo_results # Mock inference call

        # Mock EasyOCR results
        mock_easyocr_reader.return_value.readtext.return_value = [
            ([ [50, 60], [150, 60], [150, 100], [50, 100] ], "TEST1234", 0.8)  # Mock OCR result
        ]

        result = alpr_processor.process_frame(dummy_image, mock_db_conn) # Pass mock db connection

        self.assertIsNotNone(result)
        self.assertEqual(result['plate_number'], 'TEST1234')
        mock_db_conn.insert_plate_data.assert_called_once()

    @patch('src.alpr.YOLO')  # Mock yolo
    @patch('src.alpr.easyocr.Reader')  # Mock EasyOCR
    @patch('src.alpr.db.connect_to_db') # Mock database connection
    def test_process_frame_no_detection(self, mock_connect_db, mock_easyocr, mock_yolo):
        """Test the case where no license plate is detected."""
        mock_db_conn = MagicMock() # Mock database connection object.
        mock_connect_db.return_value = mock_db_conn # Make connect_to_db return the mock.
        alpr_processor = src.alpr.ALPRProcessor(test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock YOLO results (no detection)
        mock_yolo_results = [MagicMock()]
        mock_yolo_results[0].boxes.cpu.return_value.numpy.return_value = []  # Empty detection list
        mock_yolo.return_value.return_value = mock_yolo_results # Mock inference call
        result = alpr_processor.process_frame(dummy_image, mock_db_conn) # Pass mock db connection
        self.assertIsNone(result)
        mock_easyocr.assert_not_called()  # EasyOCR shouldn't be called
        mock_db_conn.insert_plate_data.assert_not_called()


    @patch('src.alpr.YOLO')  # Mock yolo
    @patch('src.alpr.easyocr.Reader')  # Mock EasyOCR
    @patch('src.alpr.db.connect_to_db') # Mock database connection
    def test_process_frame_ocr_failure(self, mock_connect_db, mock_easyocr_reader, mock_yolo):
        """Test the case where OCR fails to extract text."""
        mock_db_conn = MagicMock() # Mock database connection object.
        mock_connect_db.return_value = mock_db_conn # Make connect_to_db return the mock.
        alpr_processor = src.alpr.ALPRProcessor(test_config)
        dummy_image = np.zeros((100, 200, 3), dtype=np.uint8)

        # Mock YOLO results
        mock_yolo_results = [MagicMock()]
        mock_yolo_results[0].boxes.cpu.return_value.numpy.return_value = [
            MagicMock(cls=[2], xyxy=[[50, 60, 150, 100]])  # Simulate coordinates
        ]
        mock_yolo_results[0].names = {2: 'car'}
        mock_yolo.return_value.return_value = mock_yolo_results  # Mock inference call

        # Mock EasyOCR results (empty result)
        mock_easyocr_reader.return_value.readtext.return_value = [] # Empty result

        result = alpr_processor.process_frame(dummy_image, mock_db_conn) # Pass mock db connection
        self.assertIsNone(result)
        mock_db_conn.insert_plate_data.assert_not_called()

if __name__ == '__main__':
    unittest.main()