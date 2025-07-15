import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def load_video(video_path: str):
    """
    Loads a video file and returns a VideoCapture object.
    Args:
        video_path (str): Path to the video file.
    Returns:
        cv2.VideoCapture: VideoCapture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    return cap

def create_video_writer(output_path: str, cap: cv2.VideoCapture, fps: float = None):
    """
    Creates a VideoWriter object for saving the processed video.
    Args:
        output_path (str): Path where the output video will be saved.
        cap (cv2.VideoCapture): The input VideoCapture object to get frame properties.
        fps (float, optional): Frames per second for the output video. Defaults to input video's FPS.
    Returns:
        cv2.VideoWriter: VideoWriter object if successful, None otherwise.
    """
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return None
    return out

# Global variables for feature extraction model and transforms
_feature_extractor_model = None
_feature_extractor_transform = None

def initialize_feature_extractor():
    """
    Initializes the pre-trained ResNet50 model for feature extraction.
    Loads the model to GPU if available, otherwise runs on CPU.
    """
    global _feature_extractor_model, _feature_extractor_transform
    if _feature_extractor_model is None:
        print("Initializing feature extractor (ResNet50)...")
        # Load a pre-trained ResNet50 model, remove the last classification layer
        model = models.resnet50(pretrained=True)
        _feature_extractor_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        _feature_extractor_model.eval() # Set to evaluation mode

        # Move model to GPU if available
        if torch.cuda.is_available():
            _feature_extractor_model = _feature_extractor_model.cuda()
            print("Feature extractor moved to GPU.")
        else:
            print("Feature extractor running on CPU.")
        
        # Define transformations for the input images (expected by ResNet)
        _feature_extractor_transform = transforms.Compose([
            transforms.Resize((224, 224)), # ResNet expects 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Feature extractor initialized.")

def extract_features(image_patch: np.ndarray):
    """
    Extracts features from an image patch (e.g., a player's bounding box).
    Args:
        image_patch (np.ndarray): A NumPy array representing the cropped image (H, W, C).
    Returns:
        np.ndarray: A 1D NumPy array of extracted features.
    """
    if _feature_extractor_model is None or _feature_extractor_transform is None:
        initialize_feature_extractor()

    # Convert NumPy array (BGR) to PIL Image (RGB) for torchvision transforms
    pil_image = Image.fromarray(cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
    
    # Apply transformations and add batch dimension (unsqueeze)
    input_tensor = _feature_extractor_transform(pil_image).unsqueeze(0)
    
    # Move tensor to GPU if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Perform inference without computing gradients
    with torch.no_grad():
        features = _feature_extractor_model(input_tensor)
    
    # Flatten the features and convert to NumPy array on CPU
    return features.cpu().numpy().flatten()

def draw_bboxes(frame: np.ndarray, detections: list, ball_id_val: int = None, 
                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 0.7, thickness: int = 2):
    """
    Draws bounding boxes and labels on the frame.
    Args:
        frame (np.ndarray): The image frame to draw on.
        detections (list): List of detections, each being (x1, y1, x2, y2, confidence, class_id, track_id).
        ball_id_val (int, optional): The fixed ID used for the main ball. Defaults to None.
        font: OpenCV font type.
        font_scale (float): Scale of the font.
        thickness (int): Thickness of lines and text.
    Returns:
        np.ndarray: The frame with bounding boxes and labels drawn.
    """
    # Define Class IDs from YOLO Model (ensure these match main.py for consistency)
    BALL_CLASS_ID = 0
    GOALKEEPER_CLASS_ID = 1
    PLAYER_CLASS_ID = 2
    REFEREE_CLASS_ID = 3

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        confidence = det[4]
        class_id = int(det[5])
        track_id_raw = det[6] if len(det) > 6 else None # Can be int or string

        # Determine the color and label prefix based on class_id
        if class_id == BALL_CLASS_ID:
            color = (0, 165, 255) # Orange for ball
            label_prefix = "Ball"
        elif class_id == GOALKEEPER_CLASS_ID:
            color = (255, 0, 0) # Blue for goalkeeper
            label_prefix = "GK"
        elif class_id == PLAYER_CLASS_ID:
            color = (0, 255, 0) # Green for player
            label_prefix = "Player"
        elif class_id == REFEREE_CLASS_ID:
            color = (0, 255, 255) # Yellow for referee
            label_prefix = "Ref"
        else: # Fallback for unknown classes
            color = (128, 128, 128) # Grey
            label_prefix = "Unknown"

        # Format the display label for track ID
        if track_id_raw is not None:
            if ball_id_val is not None and track_id_raw == ball_id_val:
                label_id = "Main" # Display "Main" for the primary ball
            elif isinstance(track_id_raw, (int, np.integer)):
                label_id = str(track_id_raw) # Convert integer IDs to string
            else:
                label_id = str(track_id_raw) # Display string IDs directly
            label = f"{label_prefix} ID: {label_id}"
        else: # No ID assigned
            label = f"{label_prefix} Class: {class_id}"

        label += f" Conf: {confidence:.2f}"

        # Draw rectangle and put text label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame