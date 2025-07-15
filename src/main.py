import os
import sys
import cv2

# Add the project root to the system path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ultralytics import YOLO
from src.utils import load_video, draw_bboxes, create_video_writer, extract_features, initialize_feature_extractor
import faiss
import numpy as np

# Define Class IDs from YOLO Model
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Define Paths
VIDEO_PATH = os.path.join('data', 'input', '15sec_input_720p.mp4')
YOLO_MODEL_PATH = os.path.join('src', 'models', 'yolov11s.pt')
OUTPUT_VIDEO_PATH = os.path.join('output', '15sec_output_tracked.mp4') # Renamed output file

# Global variables for tracking
next_track_id = 0
player_faiss_index = None # Stores player and goalkeeper features
faiss_to_track_id_map = [] # Maps FAISS internal index to our assigned track_id
FAISS_DIM = 2048 # ResNet50 features are 2048-dimensional

# Re-identification Threshold (Adjusted for best observed performance)
REID_THRESHOLD = 100.0 

# Global for ball tracking
ball_track_id = -1 # Fixed ID for the main ball

def get_next_track_id():
    global next_track_id
    current_id = next_track_id
    next_track_id += 1
    return current_id

def run_tracking_pipeline():
    global player_faiss_index, next_track_id, faiss_to_track_id_map, ball_track_id

    print("Initializing components for player re-identification...")

    initialize_feature_extractor()

    # Initialize FAISS index for players and goalkeepers
    player_faiss_index = faiss.IndexFlatL2(FAISS_DIM)
    print(f"FAISS index initialized for players/goalkeepers (Dim: {FAISS_DIM}).")

    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"Error: YOLOv11 model not found at {YOLO_MODEL_PATH}.")
            print("Please download it from https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrpPvcMD/view")
            print("and place it in the 'src/models/' directory.")
            return
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLOv11 model loaded successfully.")
        print("YOLO Model Class Names:", model.names)
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        return

    cap = load_video(VIDEO_PATH)
    if cap is None:
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video '{os.path.basename(VIDEO_PATH)}' ({total_frames} frames at {fps:.2f} FPS).")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    out_writer = create_video_writer(OUTPUT_VIDEO_PATH, cap)
    if out_writer is None:
        cap.release()
        return

    frame_count = 0
    # Store active track IDs and their last seen frame for potential future use (e.g., pruning stale tracks)
    active_tracked_ids = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete or error reading frame.")
            break

        frame_count += 1
        # print(f"Processing frame {frame_count}/{total_frames}...") # Can uncomment for detailed progress

        results = model(frame, verbose=False)[0]
        
        current_frame_detections_for_drawing = []
        ball_detected_in_this_frame = False 
        
        player_gk_detections_in_frame = []
        other_detections_in_frame = []

        # Separate detections into categories
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            current_bbox = [x1, y1, x2, y2]
            class_id_int = int(class_id)

            if class_id_int == BALL_CLASS_ID:
                # Assign fixed ID for the primary ball, unique IDs for extra balls
                if not ball_detected_in_this_frame:
                    assigned_id = ball_track_id
                    ball_detected_in_this_frame = True
                else:
                    assigned_id = f"Ball_Extra_{get_next_track_id()}"
                other_detections_in_frame.append(current_bbox + [score, class_id_int, assigned_id])
            
            elif class_id_int in [GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID]:
                player_gk_detections_in_frame.append({'bbox': current_bbox, 'score': score, 'class_id': class_id_int})
            
            elif class_id_int == REFEREE_CLASS_ID:
                # Referees get unique IDs for drawing, not re-identified based on features
                assigned_id = f"Ref_{get_next_track_id()}"
                other_detections_in_frame.append(current_bbox + [score, class_id_int, assigned_id])

        # Process Players and Goalkeepers for Re-identification
        for p_det in player_gk_detections_in_frame:
            x1, y1, x2, y2 = p_det['bbox']
            score = p_det['score']
            class_id = p_det['class_id']

            # Ensure valid bounding box crop
            x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
            x2_c, y2_c = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            assigned_id = None # Reset assigned_id for each detection

            if x2_c > x1_c and y2_c > y1_c:
                player_patch = frame[y1_c:y2_c, x1_c:x2_c]
                
                if player_patch.size > 0:
                    features = extract_features(player_patch)

                    # Query FAISS index for nearest neighbor
                    if player_faiss_index.ntotal > 0: 
                        D, I = player_faiss_index.search(np.array([features]).astype('float32'), 1)
                        distance = D[0][0]
                        retrieved_faiss_idx = I[0][0]

                        # Assign existing ID if match found within threshold
                        if retrieved_faiss_idx != -1 and distance < REID_THRESHOLD:
                            assigned_id = faiss_to_track_id_map[retrieved_faiss_idx]
                            # Update last seen frame for the re-identified track
                            active_tracked_ids[assigned_id] = {'last_seen_frame': frame_count, 'class_id': class_id}
                    
                    # If no existing match, assign a new ID and add to FAISS
                    if assigned_id is None:
                        assigned_id = get_next_track_id()
                        player_faiss_index.add(np.array([features]).astype('float32'))
                        faiss_to_track_id_map.append(assigned_id)
                        active_tracked_ids[assigned_id] = {'last_seen_frame': frame_count, 'class_id': class_id}
                    
                    current_frame_detections_for_drawing.append(p_det['bbox'] + [score, class_id, assigned_id])
                else: 
                    # Handle empty patch (e.g., very small detection)
                    assigned_id = f"Invalid_{get_next_track_id()}"
                    current_frame_detections_for_drawing.append(p_det['bbox'] + [score, class_id, assigned_id])
            else: 
                # Handle invalid bounding box coordinates
                assigned_id = f"Invalid_{get_next_track_id()}"
                current_frame_detections_for_drawing.append(p_det['bbox'] + [score, class_id, assigned_id])

        # Add other detections (ball, referee) to the drawing list
        current_frame_detections_for_drawing.extend(other_detections_in_frame)

        # Draw bounding boxes and write frame to output video
        annotated_frame = draw_bboxes(frame.copy(), current_frame_detections_for_drawing, ball_id_val=ball_track_id)
        out_writer.write(annotated_frame)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to '{OUTPUT_VIDEO_PATH}'.")
    print(f"Total unique track IDs assigned (players/goalkeepers/temp): {next_track_id}")
    print(f"Total feature vectors in FAISS index (players/goalkeepers): {player_faiss_index.ntotal}")
    print(f"Number of active tracked players/goalkeepers at end: {len(active_tracked_ids)}")

if __name__ == "__main__":
    run_tracking_pipeline()