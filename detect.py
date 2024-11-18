from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")


def detectImage(path):
    results = model(path)

    print(f"{results} results")
    results[0].show()





def DetectVideo(path):
    cap = cv2.VideoCapture(path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Create a VideoWriter object to save the output video
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the FPS of the input video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break  # Exit loop if video ends

        # Run YOLO inference on the frame
        results = model(frame)  # Perform detection on the current frame

        # Results will be a list of detections (bounding boxes, class IDs, etc.)
        # Display the results (draw bounding boxes, labels, etc.)
        frame_with_boxes = results[0].plot()  # Visualize detection results on the frame

        # Display the frame with bounding boxes
        cv2.imshow("YOLO Object Detection", frame_with_boxes)

        # Write the frame with bounding boxes to the output video
        out.write(frame_with_boxes)

        # Break on key press (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# image pass video url or absolute path
DetectVideo(r"C:\Users\rozod\OneDrive\Desktop\Furniture.mp4")
# DetectVideo(0)

# for video camera 
# DetectVideo(0)

# image pass image url or absolute path

# detectImage("https://www.alankaram.in/wp-content/uploads/2022/12/A7402720-2048x1365-1.jpg")