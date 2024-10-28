import json
import os
import sys
import glob
import numpy as np
import subprocess
import re
import cv2
import piexif
import math
import tkinter as tk
import torch
import requests
import platform
import csv
from geopy.distance import geodesic
from PIL import Image
from zoedepth.utils.misc import save_raw_16bit, colorize
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from tkinter import filedialog
from tkinter import messagebox
from os.path import exists
from ultralytics import YOLO

middle_coordinates = {} 
depth_values = {}

class KalmanFilter:
    def __init__(self, process_var=1e-3, measurement_var=1e-3):
        # Initialize Kalman Filter with process and measurement variances
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.state = np.zeros((4, 1))  # State vector: [x, y, vx, vy]
        self.covariance = np.eye(4)    # Covariance matrix

    def predict(self, dt):
        # Predict the state ahead in time using constant velocity model
        transition_matrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Update state and covariance
        self.state = np.dot(transition_matrix, self.state)
        self.covariance = np.dot(np.dot(transition_matrix, self.covariance), transition_matrix.T) + self.process_var * np.eye(4)

    def update(self, measurement):
        # Update the state based on measurement using Kalman gain
        measurement_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        kalman_gain = np.dot(np.dot(self.covariance, measurement_matrix.T),
                             np.linalg.inv(np.dot(np.dot(measurement_matrix, self.covariance), measurement_matrix.T)
                                           + self.measurement_var * np.eye(2)))
        residual = measurement - np.dot(measurement_matrix, self.state)
        self.state = self.state + np.dot(kalman_gain, residual)
        self.covariance = np.dot((np.eye(4) - np.dot(kalman_gain, measurement_matrix)), self.covariance)

def show_error_message(message):
        messagebox.showerror("Error", message)
    
def show_warning_message(message):
    messagebox.showerror("Warning", message)

def show_message(message):
    messagebox.showinfo("Success", message)

def create_gui():
    def browse_button(entry_widget):
        # Function to browse directories and insert path into entry widget
        filename = filedialog.askdirectory()
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, filename)

    # Create the GUI for user interaction
    root = tk.Tk()
    root.title("Function Runner")

    # Entry widgets for file locations
    images_label = tk.Label(root, text="Images Directory:")
    images_label.grid(row=0, column=0, padx=5, pady=5)
    images_entry = tk.Entry(root, width=50)
    images_entry.grid(row=0, column=1, padx=5, pady=5)
    images_button = tk.Button(root, text="Browse", command=lambda: browse_button(images_entry))
    images_button.grid(row=0, column=3, padx=5, pady=5)

    json_label = tk.Label(root, text="JSON Directory:")
    json_label.grid(row=1, column=0, padx=5, pady=5)
    json_entry = tk.Entry(root, width=50)
    json_entry.grid(row=1, column=1, padx=5, pady=5)
    json_button = tk.Button(root, text="Browse", command=lambda: browse_button(json_entry))
    json_button.grid(row=1, column=3, padx=5, pady=5)

    ob_label = tk.Label(root, text="Object Detection Directory:")
    ob_label.grid(row=2, column=0, padx=5, pady=5)
    ob_entry = tk.Entry(root, width=50)
    ob_entry.grid(row=2, column=1, padx=5, pady=5)
    ob_button = tk.Button(root, text="Browse", command=lambda: browse_button(ob_entry))
    ob_button.grid(row=2, column=3, padx=5, pady=5)

    # Entry widget and button for depth_estimation function
    de_label = tk.Label(root, text="Depth Estimation Directory:")
    de_label.grid(row=3, column=0, padx=5, pady=5)
    de_entry = tk.Entry(root, width=50)
    de_entry.grid(row=3, column=1, padx=5, pady=5)
    de_button = tk.Button(root, text="Browse", command=lambda: browse_button(de_entry))
    de_button.grid(row=3, column=3, padx=5, pady=5)

    # Checkbox for setting a flag
    kalman_flag = tk.IntVar()
    flag_checkbox = tk.Checkbutton(root, text="Use Kalman Filter?", variable=kalman_flag)
    flag_checkbox.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    crop_flag = tk.IntVar()
    flag_checkbox = tk.Checkbutton(root, text="Crop Images?", variable=crop_flag)
    flag_checkbox.grid(row=5, column=1, columnspan=3, padx=5, pady=5)

    # Buttons to run individual functions
    detect_button = tk.Button(root, text="Detect Objects", command=lambda: detect_objects(images_entry.get(), json_entry.get(), ob_entry.get(), crop_flag.get()))
    detect_button.grid(row=6, column=1, padx=5, pady=5)

    extract_button = tk.Button(root, text="Extract Metadata", command=lambda: extract_metadata_from_images(images_entry.get(), json_entry.get()))
    extract_button.grid(row=6, column=0, padx=5, pady=5)

    de_button = tk.Button(root, text="Depth Estimation", command=lambda: depth_estimation(images_entry.get(), de_entry.get()))
    de_button.grid(row=6, column=2, padx=5, pady=5)

    gps_trans_button = tk.Button(root, text="Translate GPS", command=lambda: calculate_new_gps_coordinate(json_entry.get(), images_entry.get(), kalman_flag.get()))
    gps_trans_button.grid(row=6, column=3, padx=5, pady=5)

    # Create a new window for displaying log messages
    log_window = tk.Toplevel()
    log_window.title("Log Window")

    # Create a text box widget for displaying log messages
    global log_text
    log_text = tk.Text(log_window, height=40, width=100)
    log_text.pack()

    global os_type
    os_type = get_os(log_text)

    root.mainloop()

def extract_metadata_from_images(images_dir, output_dir):
    # Extract metadata from images and save as JSON files
    if not os.path.exists(images_dir):
        show_error_message("No Image directory chosen. Please choose one.")
        return
    
    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))
    
    if not image_paths:
        show_error_message("No images in path.")
        return

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception:
        show_warning_message("No JSON directory specified. Using default path JSON_output")
        output_dir = "./JSON_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        json_file_path = os.path.join(output_dir, f"{image_name}.json")
        with open(json_file_path, "w") as f:
            if os_type == "Linux":
                subprocess.call(["exiftool/linux/exiftool", image_path, "-j"], stdout=f)
            elif os_type =="Windows":
                subprocess.call(["exiftool/windows/exiftool.exe", image_path, "-json"], stdout=f)
    show_message("All data processed.")

def kalman_filter(extracted_data):
    # Apply Kalman filter to extracted GPS and speed data
    kf = KalmanFilter()  # Create Kalman filter instance

    result = []
    for data in extracted_data:
        # Process GPS and speed data using Kalman filter
        if data[1] is not None and data[2] is not None and data[3] is not None and data[4] is not None and data[5] is not None:
            measurement = np.array([[data[1]], [data[2]]])
            kf.update(measurement)  # Update with GPS data
            kf.predict(1)  # Predict for next time step
            estimated_gps_latitude, estimated_gps_longitude = kf.state[:2].flatten()  # Estimated GPS coordinates
            result.extend([[data[0], estimated_gps_latitude, estimated_gps_longitude, data[3], data[4], data[5]]])

            log_text.insert(tk.END, f"Image: {data[0]} GPS Latitude: {estimated_gps_latitude}, GPS Longitude: {estimated_gps_longitude}, "
                  f"FlightXSpeed: {data[3]}, FlightYSpeed: {data[4]}, Bearing Degree: {data[5]}\n")
        else:
            log_text.insert(tk.END, f"Image {data[0]}: GPS or Speed data not found.\n")
        
    return result

def extract_data_from_json(json_file):
    # Extract relevant data from JSON file
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        result = []
        file_name = data[0].get("FileName", None)
        gps_latitude = data[0].get("GPSLatitude", None)
        gps_longitude = data[0].get("GPSLongitude", None)
        flight_x_speed = data[0].get("FlightXSpeed", None)
        flight_y_speed = data[0].get("FlightYSpeed", None)
        gimbal_yaw_degree = data[0].get("GimbalYawDegree", None)
        fov = data[0].get("FOV", None)
        
        # Extract numerical value from GPS coordinates strings
        gps_latitude = extract_numerical_value(gps_latitude)
        gps_longitude = extract_numerical_value(gps_longitude)

        if gps_longitude != abs(gps_longitude):
            gimbal_yaw_degree = -gimbal_yaw_degree

        result.extend([file_name, gps_latitude, gps_longitude, flight_x_speed, flight_y_speed, gimbal_yaw_degree, fov])
        return result
    except Exception:
        log_text.insert(tk.END, f"Error processing {json_file}/n")
        return [None, None, None, None, None, None, None]

def extract_numerical_value(coord_string):
    # Extract numerical value from GPS coordinates strings
    match = re.match(r"(-?\d+) deg (\d+)' (\d+\.\d+)\"", coord_string)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        decimal_degrees = degrees + minutes / 60 + seconds / 3600
        if "W" in coord_string:
            decimal_degrees = -decimal_degrees
        elif "S" in coord_string:
            decimal_degrees = -decimal_degrees
        return decimal_degrees
    else:
        return None
    
def sort_flight_data_by_file_name(flight_data):
    # Sort flight data based on file name
    return sorted(flight_data, key=lambda x: x[0])

def download_ob_model(model_name):
    link = "https://download1523.mediafire.com/zoz4yjgbtkbgCzMLLeeDZRQyFtJGO5SuJe2-gDz-f0g8yOx8_TS3Hm9IpAvAb_LCVHHSzLYTnwY0rqZTxj5BM90zDWjTkI_LGZBBYVWDDZMz5xPqhzEDdBwFaTECtjmHzJ0oaNmbBiOdi40EKbxyl1wKajZOQevvkWuyIASXxEk/ry3uuvjt175wfui/best.pt"
    try:
        with open(model_name, "wb") as f:
            log_text.insert(tk.END, "Downloading %s\n" % model_name)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None: # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()
                
        log_text.insert(tk.END, "\n")
    except Exception:
        show_error_message("Unable to download object detection model.")
        log_text.insert("Unable to download object detection model. This is most likely due to a dead link.\n")
        return

def detect_objects(images_dir, json_directory, output_dir, crop_flag):
    # Detect objects in images and save annotated images
    if not os.path.exists(images_dir):
        show_error_message("No Image directory chosen. Please choose one.")
        return
    
    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))
    
    if not image_paths:
        show_error_message("No images in path.")
        return

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception:
        show_warning_message("No Object Detection Directory specified. Using default path object_detection_output")
        output_dir = "./object_detection_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if json_directory == '' or json_directory == './JSON_output':
        show_warning_message("No JSON directory specified. Using default path JSON_output")
        json_directory = "./JSON_output"
        
    if not glob.glob(os.path.join(json_directory, "*.json")):
        show_error_message("No json files in path. Please run extract metadata.")
        return

    model_name = 'best.pt'

    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))

    all_flight_data = []
    
    for json_file in glob.glob(os.path.join(json_directory, "*.json")):
        data = extract_data_from_json(json_file)
        all_flight_data.append(data)

    if not exists(model_name):
        show_error_message("Model not present. Please download the model.")
        return

    # Model
    model = YOLO(model_name)

    # Object classes
    class_names = ['crossarm', 'cutouts', 'insulator', 'pole', 'transformers', 'background_structure']

    for image_path in image_paths:
        # Load the image
        img = cv2.imread(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]

        # Open the image using PIL to access EXIF data
        pil_img = Image.open(image_path)
        exif_dict = piexif.load(pil_img.info.get('exif'))

        fov = get_fov(all_flight_data, os.path.basename(image_path))
        original_width = img.shape[1]
        original_height = img.shape[0]

        # Object detection
        results = model(img)

        # Coordinates and drawing bounding boxes
        pole_coordinates = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # class name
                cls = int(box.cls[0])

                if class_names[cls] == 'pole':  # Check if the detected object is a pole
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100

                    if confidence > .7:
                        # Calculate middle point of the bottom boundary
                        middle_x = (x1 + x2) // 2
                        middle_y = y2  # Use y2 which represents the bottom boundary

                        # Store middle point coordinates
                        pole_coordinates.append((middle_x, middle_y))

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 3
                        color = (255, 0, 0)
                        thickness = 3

                        if not crop_flag:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cv2.putText(img, f"{class_names[cls]}: {confidence:.2f}", org, font, font_scale, color, thickness)

                        # Print middle point coordinates
                        log_text.insert(tk.END, f"Middle Point Coordinates for Pole: ({middle_x}, {middle_y})\n")
                        if crop_flag:
                            # Crop the image using the bounding box coordinates
                            img = img[y1:y2, x1:x2]

                            # Calculate scale factors
                            scale_x = img.shape[1] / original_width
                            scale_y = img.shape[0] / original_height

                            # Adjust FOV based on scale factors
                            adjusted_fov_x = fov * scale_x
                            adjusted_fov_y = fov * original_height/original_width * scale_y

                            # Update image width and height in EXIF data
                            exif_dict["Exif"][piexif.ExifIFD.PixelXDimension] = img.shape[1]
                            exif_dict["Exif"][piexif.ExifIFD.PixelYDimension] = img.shape[0]

                            log_text.insert(tk.END, f"Original FOV: ({fov}, {fov * original_height/original_width})\n")
                            log_text.insert(tk.END, f"Adjusted FOV: ({adjusted_fov_x}, {adjusted_fov_y})\n")

        middle_coordinates[image_path] = pole_coordinates
                    
        log_text.insert(tk.END, f"Crop Flag: {crop_flag}\n")\
        
        if crop_flag:
            output_path = os.path.join(output_dir, f"{img_name}_cropped.jpg")
        else:
            output_path = os.path.join(output_dir, f"{img_name}.jpg")

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        exif_bytes = piexif.dump(exif_dict)
        pil_img.save(output_path, exif=exif_bytes)

        log_text.insert(tk.END, f"Processed: {image_path}\n")
    show_message("All images processed successfully.")

def depth_estimation(images_dir, output_dir):
    if not os.path.exists(images_dir):
        show_error_message("No Image directory chosen. Please choose one.")
        return
    
    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))
    
    if not image_paths:
        show_error_message("No images in path.")
        return
    
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception:
        show_warning_message("No Depth Estimation Directory specified. Using default path depth_output")
        output_dir = "./depth_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if not middle_coordinates:
        show_error_message("No middle coordinates calculated. Please confirmed object detection was run.")
        return

    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))

    # Initialize the model
    conf = get_config("zoedepth_nk", "infer")
    model_zoe_nk = build_model(conf)

    # Use GPU if available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)

    # Process each image
    for image_path in image_paths:
        # Check if the input image path exists as a key in middle_coordinates
        if image_path in middle_coordinates:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Perform inference
            depth_numpy = zoe.infer_pil(image)

            # List to store depth values for the current image
            image_depth_values = []

            # Access value at middle coordinates
            for middle_coord in middle_coordinates[image_path]:
                middle_x, middle_y = middle_coord
                depth_value = depth_numpy[middle_y, middle_x]

                log_text.insert(tk.END, f"Depth value at middle coordinate ({middle_x}, {middle_y}): {depth_value}\n")

                # Append depth value to list
                image_depth_values.append(depth_value)

            # Save depth values for the current image
            depth_values[image_path] = image_depth_values

            # Construct output file paths
            image_name = os.path.basename(image_path)
            output_raw_path = os.path.join(output_dir, f"{image_name}_raw.png")
            output_colored_path = os.path.join(output_dir, f"{image_name}_colored.png")

            # Save raw depth image
            save_raw_16bit(depth_numpy, output_raw_path)

            # Colorize output
            colored = colorize(depth_numpy)

            # Save colored output
            Image.fromarray(colored).save(output_colored_path)
        else:
            log_text.insert(tk.END, f"No middle coordinates found for {image_path}\n")

    show_message("All images processed successfully.")

def calculate_new_gps_coordinate(json_directory, images_dir, kalman_flag):
    if not os.path.exists(images_dir):
        show_error_message("No Image directory chosen. Please choose one.")
        return
    
    image_paths = glob.glob(os.path.join(images_dir, "*.[Jj][Pp][Gg]"))
    
    if not image_paths:
        show_error_message("No images in path.")
        return
    
    if json_directory == '' or json_directory == './JSON_output':
        show_warning_message("No JSON directory specified. Using default path JSON_output")
        json_directory = "./JSON_output"
        
    if not glob.glob(os.path.join(json_directory, "*.json")):
        show_error_message("No json files in path. Please run extract metadata.")
        return
    
    if not depth_values:
        show_error_message("No depth values calculated. Please confirmed depth estimation was run.")
        return
    
    all_flight_data = []

    output_rows = []  # Store rows for CSV output

    for json_file in glob.glob(os.path.join(json_directory, "*.json")):
        data = extract_data_from_json(json_file)
        all_flight_data.append(data)

    sorted_flight_data = sort_flight_data_by_file_name(all_flight_data)

    log_text.insert(tk.END, f"Kalman Filter Flag: {kalman_flag}\n")
    if kalman_flag:
        log_text.insert(tk.END, "Using Kalman filter.\n")
        sorted_flight_data = kalman_filter(sorted_flight_data)

    for data in sorted_flight_data:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            if image_path in depth_values and data[0] in image_path:
                log_text.insert(tk.END, f"{image_name}\n")
                log_text.insert(tk.END, f"Original Lat: {data[1]}\n")
                log_text.insert(tk.END, f"Original Long: {data[2]}\n")

                for value in depth_values[image_path]:
                    log_text.insert(tk.END, f"Depth Value: {value}\n")
                    log_text.insert(tk.END, f"Bearing Degree: {data[5]}\n")

                    # Calculate the destination point using geodesic
                    destination_point = geodesic(kilometers=value / 1000).destination((data[1], data[2]), data[5])

                    # Extract latitude and longitude from the destination point
                    new_lat, new_lon = destination_point.latitude, destination_point.longitude

                    log_text.insert(tk.END, f"New Lat: {new_lat}\n")
                    log_text.insert(tk.END, f"New Long: {new_lon}\n")

                    # Append data to output rows for CSV
                    output_rows.append([image_name, new_lat, new_lon])

    # Write output to CSV file
    csv_filename = "output.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'lat', 'long'])  # Write header
        writer.writerows(output_rows)  # Write data rows

    show_message("All values processed.")

def get_fov(data, target_file_name):
    for entry in data:
        if entry[0] == target_file_name:
            return extract_and_convert_to_float(entry[-1])  # FOV is the last element in the entry
    return None  # Return None if the target_file_name is not found

def extract_and_convert_to_float(input_string):
    pattern = r'(\d+\.\d+)'  # Regular expression pattern to extract the numerical value
    match = re.search(pattern, input_string)  # Search for the pattern in the input string
    
    if match:
        value_str = match.group(1)  # Extract the matched value
        return float(value_str)  # Convert the value to a float
    else:
        return None  # Return None if no numerical value found in the input string

def get_os(log_text):
    try:
        log_text.insert(tk.END, f"{platform.system()} OS detected\n")  # Insert OS detection message into the log window
        return platform.system()
    except Exception:
        log_text.insert(tk.END, "Unknown OS detected. Can not continue.\n")
    
    return platform.system()

if __name__ == "__main__":
    create_gui()