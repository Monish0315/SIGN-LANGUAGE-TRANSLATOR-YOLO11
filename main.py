import cv2
from ultralytics import YOLO

# Load the model from the specified path
model = YOLO(r'C:\Users\pmoni\project\sign_language_detection\best.pt')  # Update with your saved model path

# Open the camera
cap = cv2.VideoCapture(0)

# Set camera resolution (optional, can adjust if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Start the video capture and process each frame
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run inference and get results
        results = model.track(frame, persist=True)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("ASL Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if there's an error with frame capture
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
