import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model_path = "/home/kub/workspace/cinema8/yolo-track/models/yolov9e.pt"
model = YOLO(model_path)

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the device

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): (xmin, ymin, xmax, ymax) for the first box.
        box2 (tuple): (xmin, ymin, xmax, ymax) for the second box.

    Returns:
        float: IoU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def preprocess_frame(frame):
    """
    Preprocess the frame to match the YOLO model's expected input format.
    
    Args:
        frame (np.ndarray): Input frame in HWC format.

    Returns:
        torch.Tensor: Preprocessed frame in BCHW format.
    """
    # Resize the frame to (640, 640)
    resized_frame = cv2.resize(frame, (640, 640))
    # Convert the frame from HWC to CHW format
    chw_frame = resized_frame.transpose(2, 0, 1)
    # Normalize the frame to [0, 1]
    normalized_frame = chw_frame / 255.0
    # Convert the frame from numpy array to torch tensor
    tensor_frame = torch.from_numpy(normalized_frame).float()
    # Add a batch dimension (BCHW)
    tensor_frame = tensor_frame.unsqueeze(0).to(device)
    return tensor_frame, resized_frame

def track_object(video_path, xPer, yPer, wPer, hPer, intime, outtime, expFrame):
    """
    Track an object in the video using YOLO model.

    Args:
        video_path (str): Path to the local video file.
        xPer (float): X-coordinate of the initial bounding box as a percentage.
        yPer (float): Y-coordinate of the initial bounding box as a percentage.
        wPer (float): Width of the initial bounding box as a percentage.
        hPer (float): Height of the initial bounding box as a percentage.
        intime (float): Start time of the tracking in seconds.
        outtime (float): End time of the tracking in seconds.
        expFrame (int): Frame interval for tracking.

    Returns:
        list: A list of dictionaries with tracking results.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Video could not be opened!")

    # Get video properties
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert percentage values to pixel values for initial bounding box
    x = int(xPer * video_width / 100)
    y = int(yPer * video_height / 100)
    width = int(wPer * video_width / 100)
    height = int(hPer * video_height / 100)

    # Set the start position of the video
    video.set(cv2.CAP_PROP_POS_MSEC, intime * 1000)

    results = []
    frame_count = 0

    # Initial bounding box
    prev_box = (x, y, x + width, y + height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output.mp4'
    fps = video.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret or video.get(cv2.CAP_PROP_POS_MSEC) > outtime * 1000:
            break

        if frame_count % expFrame == 0:
            # Preprocess the frame
            tensor_frame, resized_frame = preprocess_frame(frame)

            # Use YOLO model to detect objects in the frame
            detection_results = model(tensor_frame)
            detections = detection_results[0]  # Get the first result from the model output

            # Scale factor for bounding box coordinates
            scale_x = video_width / 640
            scale_y = video_height / 640

            # Find the object with the highest IoU compared to the previous bounding box
            best_iou = 0
            best_box = None
            for detection in detections.boxes:
                box = detection.xyxy[0].tolist()  # Get bounding box coordinates
                if len(box) == 4:  # Ensure we have the correct number of coordinates
                    xmin, ymin, xmax, ymax = box  # Unpack the bounding box coordinates
                    # Scale the coordinates back to original video dimensions
                    xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
                    ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)
                    current_box = (xmin, ymin, xmax, ymax)
                    iou = calculate_iou(prev_box, current_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_box = current_box

            if best_box is not None:
                xmin, ymin, xmax, ymax = best_box
                x, y, width, height = xmin, ymin, xmax - xmin, ymax - ymin
                prev_box = best_box
                # Convert to percentages for results
                x_per = x / video_width * 100
                y_per = y / video_height * 100
                width_per = width / video_width * 100
                height_per = height / video_height * 100
                results.append({
                    "x": x_per,
                    "y": y_per,
                    "width": width_per,
                    "height": height_per,
                    "time": video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                })
                # Draw the bounding box on the frame in original scale
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            else:
                results.append({
                    "x": None,
                    "y": None,
                    "width": None,
                    "height": None,
                    "time": video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                })

        # Write the frame with the bounding box
        out.write(frame)
        frame_count += 1

    video.release()
    out.release()
    cv2.destroyAllWindows()

    return results