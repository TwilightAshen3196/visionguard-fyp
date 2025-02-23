import unittest
import os
import src.database as db
import configparser
import sqlite3

#Create a dummy config file for testing.
test_config = configparser.ConfigParser()
test_config['Database'] = {'DatabasePath': 'test_alpr_data.db'}
test_config['Logging'] = {} #Not used.
test_config['API'] = {} #Not used
test_config['Camera'] = {} #Not used.
class TestDatabaseConnection(unittest.TestCase):
    def setUp(self):
        """Setup method to create a test database connection."""
        self.db_path = test_config['Database']['DatabasePath']
        self.db_conn = db.DatabaseConnection(self.db_path)
        self.db_conn.connect()

    def tearDown(self):
        """Teardown method to close the connection and remove the test database."""
        self.db_conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


    def test_create_tables(self):
        """Test if the table is created successfully."""
        # Connect again to make sure it does not raise error when creating exist tables.
        self.db_conn = db.DatabaseConnection(self.db_path)
        self.db_conn.connect() # This should not raise any errors.
        # Check if the table exists
        cursor = self.db_conn.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='license_plates'")
        self.assertIsNotNone(cursor.fetchone())

    def test_insert_plate_data(self):
        test_data = {
            'plate_number': 'TEST1234',
            'image_path': 'test_image.jpg',
            'detection_time': '2024-07-27 12:00:00',
            'location': 'Test Location',
            'user_id': 'test_user'
        }
        inserted_id = self.db_conn.insert_plate_data(test_data)
        self.assertIsNotNone(inserted_id)

        cursor = self.db_conn.conn.cursor()
        # Retrieve by the returned id.
        cursor.execute("SELECT * FROM license_plates where id = ?", (inserted_id,))
        retrieved_data = cursor.fetchone()

        self.assertIsNotNone(retrieved_data)
        self.assertEqual(retrieved_data[1], 'TEST1234')
        self.assertEqual(retrieved_data[2], 'test_image.jpg')
        self.assertEqual(retrieved_data[3], '2024-07-27 12:00:00')
        self.assertEqual(retrieved_data[4], 'Test Location')
        self.assertEqual(retrieved_data[5], 'test_user')


    def test_connect_to_db_function(self):
        """Test the global connect_to_db function."""
        # Clean up previous db.
        self.db_conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        conn = db.connect_to_db() # Call the function
        self.assertIsInstance(conn, db.DatabaseConnection) #Check return value type.
        self.assertIsInstance(conn.conn, sqlite3.Connection)  # Check connection.

        #Close the connection.
        conn.close()


if __name__ == '__main__':
    unittest.main()