import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import src.alpr as alpr
import src.database as db
import src.utils as utils
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

class ALPRApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = int(config['Camera']['CameraIndex'])
        self.vid = None  # Initialize to None
        self.db_conn = db.connect_to_db()
        self.alpr_processor = alpr.ALPRProcessor(self.db_conn, config)
        self.canvas = None  # Initialize canvas to None

        # --- GUI Elements ---
        self.btn_snapshot = ttk.Button(window, text="Snapshot", command=self.snapshot, state=tk.DISABLED)  # Initially disabled
        self.btn_snapshot.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_load_image = ttk.Button(window, text="Load Image", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_load_video = ttk.Button(window, text="Load Video", command=self.load_video)  # New button
        self.btn_load_video.pack(side=tk.LEFT, padx=5, pady=5)

        self.log_text = tk.Text(window, height=10, width=80)
        self.log_text.pack(pady=5)
        self.log_text.config(state=tk.DISABLED)

        self.delay = 15
        self.is_video_processing = False # Add a flag

        # Try to open the video source, handle errors gracefully
        try:
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", self.video_source)

            # Create canvas *only* if video capture is successful
            self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.canvas.pack()
            self.btn_snapshot.config(state=tk.NORMAL)  # Enable snapshot button
            self.update() # Start the update loop only if the camera is initialized.

        except Exception as e:
            utils.log_message(f"Camera initialization failed: {e}", level="ERROR")
            self.vid = None  # Ensure vid is None on failure
            utils.show_error(window, f"Camera Error: {e}")  # Use show_error for consistency



    def snapshot(self):
        if self.vid:
            ret, frame = self.vid.read()
            if ret:
                cv2.imwrite("snapshot-" + time.strftime("%Y%m%d-%H%M%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Could not open or read image at {file_path}")
                threading.Thread(target=self.process_frame, args=(image,)).start()
            except Exception as e:
                utils.log_message(f"Error loading image: {e}", level="ERROR")
                self.update_log(f"Error loading image: {e}")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]) #Added supported extensions
        if file_path:
            try:
                if self.vid and self.vid.isOpened(): # Close current video if processing.
                  self.vid.release()
                self.vid = cv2.VideoCapture(file_path)
                if not self.vid.isOpened():
                    raise ValueError(f"Could not open or read video at {file_path}")

                #Update canvas
                if self.canvas is None: # If the canvas haven't been created (no camera)
                    self.canvas = tk.Canvas(self.window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.canvas.pack()

                # Set video processing
                self.is_video_processing = True
                # Start a single thread.
                threading.Thread(target=self.process_video).start()

            except Exception as e:
                utils.log_message(f"Error loading video: {e}", level="ERROR")
                self.update_log(f"Error loading video: {e}")
                utils.show_error(self.window, f"Video Load Error: {e}") # Display the error


    def process_video(self):
        while self.is_video_processing:  # Use the flag for control
            ret, frame = self.vid.read()
            if ret:
                self.process_frame(frame)
                # Update the canvas
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if self.canvas: # Check if canvas exists.
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                time.sleep(self.delay / 1000)  # Control frame rate
            else:
                # End of video (or error).
                self.is_video_processing = False # Reset the flag
                self.vid.release()
                self.vid = None
                break  # Exit the loop


    def update(self):
        if self.vid:  # Only update if self.vid is not None
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if self.canvas:
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                threading.Thread(target=self.process_frame, args=(frame.copy(),)).start()

        #Always schedule the next update check, but only camera will update the frame.
        self.window.after(self.delay, self.update)



    def process_frame(self, frame):
        try:
            plate_data = self.alpr_processor.process_frame(frame)
            if plate_data:
                log_message = (f"Detected: {plate_data['plate_number']}, "
                               f"Timestamp: {plate_data['detection_time']}")
                self.update_log(log_message)
        except Exception as e:
            utils.log_message(f"Error processing frame: {e}", level="ERROR")
            self.update_log(f"Error processing frame: {e}")

    def update_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def on_closing(self):
        self.is_video_processing = False # Set to stop.
        if self.vid and self.vid.isOpened():
            self.vid.release()
        if self.db_conn:
            self.db_conn.close()
        self.window.destroy()
        utils.log_message("Application closed.")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ALPRApp(root, "VisionGuard ALPR System")
    except Exception as e:
        utils.log_message(f"Critical error during startup: {e}", level="CRITICAL")
        print(f"Critical error: {e}. See log file for details.")