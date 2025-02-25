import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import time
import src.alpr as alpr
import src.database as db # Import database module, but not connection class directly here
import src.utils as utils
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

class ALPRApp:
    def __init__(self, window, window_title):
        print("DEBUG: ALPRApp.__init__ started")
        self.window = window
        self.window.title(window_title)
        print("DEBUG: GUI setup (window, title) done")


        self.alpr_processor = alpr.ALPRProcessor(config) # Database connection is no longer passed here
        print("DEBUG: alpr_processor initialized successfully")

        self.image_display_panel = None # Panel to display loaded images/videos
        print("DEBUG: image_display_panel initialized to None")

        # --- GUI Elements ---
        self.btn_load_image = ttk.Button(window, text="Load Image", command=self.load_image)
        self.btn_load_image.pack(side=tk.LEFT, padx=5, pady=5)
        print("DEBUG: Load Image button created")

        self.btn_load_video = ttk.Button(window, text="Load Video", command=self.load_video)
        self.btn_load_video.pack(side=tk.LEFT, padx=5, pady=5)
        print("DEBUG: Load Video button created")

        self.log_text = tk.Text(window, height=10, width=80)
        self.log_text.pack(pady=5)
        self.log_text.config(state=tk.DISABLED)
        print("DEBUG: Log Text area created")

        self.delay = 30 # Increased delay for less CPU usage in video processing
        self.is_video_processing = False
        self.current_video_path = None # Store the path of the currently loaded video
        print("DEBUG: Delay, is_video_processing, current_video_path initialized")


        print("DEBUG: ALPRApp.__init__ finished")
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Could not open or read image at {file_path}")
                self.display_image(image) # Display the loaded image
                threading.Thread(target=self.process_image_thread, args=(image,)).start() # Process in thread
            except Exception as e:
                utils.log_message(f"Error loading image: {e}", level="ERROR")
                self.update_log(f"Error loading image: {e}")

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]) #Added supported extensions
        if file_path:
            try:
                self.current_video_path = file_path # Store the video path
                self.is_video_processing = True # Set video processing flag
                threading.Thread(target=self.process_video).start() # Start video processing thread
            except Exception as e:
                utils.log_message(f"Error loading video: {e}", level="ERROR")
                self.update_log(f"Error loading video: {e}")
                utils.show_error(self.window, f"Video Load Error: {e}") # Display the error

    def display_image(self, image):
        """Displays the given OpenCV image in the GUI."""
        try:
            if self.image_display_panel is not None: # Clear previous image if any
                self.image_display_panel.destroy()
            self.image_display_panel = tk.Label(self.window) # Create a Label for image display
            self.image_display_panel.pack()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
            image_pil = Image.fromarray(image_rgb) # Convert to PIL Image
            image_tk = ImageTk.PhotoImage(image_pil) # Convert to Tk PhotoImage
            self.image_display_panel.config(image=image_tk) # Configure the Label to display image
            self.image_display_panel.image = image_tk # Keep a reference to prevent garbage collection
        except Exception as e:
            utils.log_message(f"Error displaying image: {e}", level="ERROR")
            self.update_log(f"Error displaying image: {e}")


    def process_video(self):
        video_path = self.current_video_path
        if not video_path:
            return # Exit if no video path is set

        vid = cv2.VideoCapture(video_path) # Open video capture
        if not vid.isOpened():
            utils.log_message(f"Error: Could not open video file: {video_path}", level="ERROR")
            self.update_log(f"Error: Could not open video file: {video_path}")
            self.is_video_processing = False # Reset flag
            return

        try:
            while self.is_video_processing:  # Control loop with the flag
                ret, frame = vid.read() # Read frame
                if not ret: # End of video or error
                    break

                self.display_image(frame) # Display current frame
                self.process_frame_thread(frame.copy()) # Process frame in thread

                time.sleep(self.delay / 1000) # Control frame rate and reduce CPU usage
        finally: # Ensure resources are released even if errors occur.
            vid.release() # Release video capture
            self.is_video_processing = False # Reset flag
            self.current_video_path = None # Clear current video path
            utils.log_message("Video processing finished.")
            self.update_log("Video processing finished.")


    def process_image_thread(self, image): # Threaded function for image processing
        db_conn = None # Initialize db_conn to None
        try:
            db_conn = db.connect_to_db() # Create NEW database connection for THIS thread
            plate_data = self.alpr_processor.process_frame(frame=image, db_conn=db_conn) # Process the image, pass db_conn
            if plate_data: # If plate data is detected
                log_message = (f"Detected: {plate_data['plate_number']}, "
                               f"Timestamp: {plate_data['detection_time']}")
                self.update_log(log_message) # Update the log
        except Exception as e:
            utils.log_message(f"Error processing image in thread: {e}", level="ERROR")
            self.update_log(f"Error processing image: {e}")
        finally:
            if db_conn:
                db_conn.close() # Close the database connection in THIS thread


    def process_frame_thread(self, frame): # Threaded function for video frame processing
        db_conn = None # Initialize db_conn to None
        try:
            db_conn = db.connect_to_db() # Create NEW database connection for THIS thread
            plate_data = self.alpr_processor.process_frame(frame=frame, db_conn=db_conn) # Process frame, pass db_conn
            if plate_data:
                log_message = (f"Detected: {plate_data['plate_number']}, "
                               f"Timestamp: {plate_data['detection_time']}")
                self.update_log(log_message)
        except Exception as e:
            utils.log_message(f"Error processing frame in thread: {e}", level="ERROR")
            self.update_log(f"Error processing frame: {e}")
        finally:
            if db_conn:
                db_conn.close() # Ensure database connection is closed in THIS thread


    def update_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def on_closing(self):
        self.is_video_processing = False # Set to stop video processing loop
        if self.db_conn: # No longer instance variable, remove this line.
            pass # self.db_conn.close() # Close database connection - no longer needed here.
        self.window.destroy() # Destroy main window
        utils.log_message("Application closed.") # Log application closing


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ALPRApp(root, "VisionGuard ALPR System") # Create and run the app
    except Exception as e:
        utils.log_message(f"Critical error during startup: {e}", level="CRITICAL")
        print(f"Critical error: {e}. See log file for details.")