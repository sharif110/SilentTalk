import cv2
import os
import json
from openpose import pyopenpose as op

# Setup OpenPose parameters
params = {
    "model_folder": "path/to/openpose/models",
    "net_resolution": "-1x368",  # Change the resolution if needed
    "number_people_max": 1,  # Limit to a single person detection
    "write_json": "output/json_folder",  # Folder to save the JSON output
}

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Function to process a single frame and generate JSON data
def process_frame(frame, output_path):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Get the JSON data for the first person in the frame
    json_data = datum.poseKeypoints[0].tolist()

    # Save the JSON data to a file
    with open(output_path, "w") as f:
        json.dump(json_data, f)

# Input video file
video_path = "path/to/input/video.mp4"

# Create a directory for storing JSON files
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_directory = os.path.join("output/json_folder", video_name)
os.makedirs(output_directory, exist_ok=True)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    # Read the next frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame and generate JSON data
    output_json_path = os.path.join(output_directory, f"frame_{frame_count}.json")
    process_frame(frame, output_json_path)

    frame_count += 1

# Release the video capture and clean up
video_capture.release()
cv2.destroyAllWindows()