import logging
import time
import datetime
import configparser
import tkinter as tk
from tkinter import ttk

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# --- Configure Logging ---
log_file = config['Logging']['LogFile']
log_level_str = config['Logging']['LogLevel']
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

logging.basicConfig(filename=log_file, level=log_level,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_message(message, level="INFO"):
    if level == "DEBUG":
        logging.debug(message)
    elif level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "CRITICAL":
        logging.critical(message)
    print(f"{level}: {message}")

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def show_error(parent, message):
    """Displays an error message in a consistent way."""
    error_label = tk.Label(parent, text=message, fg="red")
    error_label.pack()