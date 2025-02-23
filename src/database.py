import sqlite3
import src.utils as utils
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

class DatabaseConnection:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.create_tables()
            utils.log_message(f"Successfully connected to database at {self.db_path}")
            return self
        except sqlite3.Error as e:
            utils.log_message(f"Error connecting to database: {e}", level="ERROR")
            raise

    def create_tables(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS license_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    image_path TEXT,
                    detection_time TEXT NOT NULL,
                    location TEXT,
                    user_id TEXT
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            utils.log_message(f"Error creating tables: {e}", level="ERROR")
            raise

    def insert_plate_data(self, plate_data):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO license_plates (plate_number, image_path, detection_time, location, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_data['plate_number'], plate_data['image_path'], plate_data['detection_time'],
                  plate_data.get('location'), plate_data.get('user_id')))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            utils.log_message(f"Error inserting data: {e}", level="ERROR")
            raise

    def close(self):
      if self.conn:
          self.conn.close()
          utils.log_message("Database connection closed.")


def connect_to_db():
    db_connection = DatabaseConnection(config['Database']['DatabasePath'])
    return db_connection.connect()