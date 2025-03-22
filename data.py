import cv2
import os

# Path to input video
video_path = "video/normal/cropped_normal_event_03.mp4"
output_folder = "frames_output"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = fps // 5  # Number of frames to skip (0.5 second interval)
frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends
    
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_normal_3_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        saved_frame_count += 1
    
    frame_count += 1

cap.release()
print("Processing complete. Frames saved in", output_folder)
