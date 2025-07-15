# Soccer Player Re-Identification Assignment

This repository contains the solution for the Player Re-Identification in Sports Footage assignment, part of the interview process for an AI Intern role at Liat.ai. The primary goal is to implement a system that can detect players in a given 15-second video clip and consistently assign them unique IDs, even if they move out of frame and re-enter.

## Table of Contents
- [Project Overview](#project-overview)
- [Features Implemented](#features-implemented)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Output](#output)
- [Challenges and Future Improvements](#challenges-and-future-improvements)
- [Contact](#contact)

## Project Overview

The assignment focuses on a real-world computer vision challenge in sports analytics: **Player Re-Identification**. The system processes a 15-second football match video (`15sec_input_720p.mp4`). It utilizes an object detection model to identify players, goalkeepers, referees, and the ball. For players and goalkeepers, a feature-based re-identification approach is implemented using FAISS for efficient similarity search to maintain consistent tracking IDs throughout the video.

## Features Implemented

* **Object Detection:** Leverages a fine-tuned Ultralytics YOLOv11 model to detect players, goalkeepers, referees, and the ball.
* **Feature Extraction:** Extracts visual features from detected player and goalkeeper bounding boxes using a pre-trained ResNet50 Convolutional Neural Network.
* **Similarity Search:** Utilizes FAISS (Facebook AI Similarity Search) to perform efficient nearest-neighbor searches, matching new detections with previously seen players based on their extracted features.
* **ID Assignment & Basic Tracking:**
    * Assigns a consistent, fixed ID for the main ball.
    * Assigns unique temporary IDs for referees and multiple/invalid ball detections.
    * Attempts to re-identify players and goalkeepers, assigning them consistent unique IDs across frames and through brief occlusions or re-entries, based on feature similarity.
* **Annotated Video Output:** Generates an output video with bounding boxes, class labels, and assigned IDs overlaid on each detected object.

## File Structure

```
player_reid_assignment/
├── data/
│   └── input/
│       └── 15sec_input_720p.mp4  # Input video for processing
├── output/
│   └── 15sec_output_tracked.mp4 # Output video with tracking
├── src/
│   ├── models/                # Directory to store the YOLOv11 model
│   │   └── yolov11s.pt        # Pre-trained YOLOv11 model (download separately)
│   ├── init.py
│   ├── main.py                # Main script for the re-identification pipeline
│   └── utils.py               # Utility functions (video I/O, drawing, feature extraction)
├── .gitignore                 # Specifies files/folders to be ignored by Git
├── README.md                  # This documentation file
└── requirements.txt           # Python dependencies
```
## Setup and Installation

Follow these steps to set up and run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GitGautamHub/player_reid_assignment.git](https://github.com/GitGautamHub/player_reid_assignment.git)
    cd player_reid_assignment
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    ```
    * **On Windows (PowerShell):**
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    * **On Windows (Command Prompt):**
        ```bash
        .\venv\Scripts\activate.bat
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Note for OpenCV:** If you encounter issues with `cv2.imshow` (e.g., "function not implemented" error), uninstall `opencv-python` and install `opencv-contrib-python` explicitly:
        ```bash
        pip uninstall opencv-python
        pip install opencv-contrib-python
        ```
    * **Note for PyTorch (GPU users):** For GPU acceleration, ensure you install the CUDA-enabled version of PyTorch. Refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for specific installation commands based on your CUDA version (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).

4.  **Download the YOLOv11 Model:**
    The assignment provides a specific fine-tuned YOLOv11 model. Please download it from the following Google Drive link:
    [https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrpPvcMD/view](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrpPvcMD/view)
    Place the downloaded `.pt` file (e.g., `yolov11s.pt`) inside the `src/models/` directory.

5.  **Place the Input Video:**
    The assignment's input video (`15sec_input_720p.mp4`) should be placed in the `data/input/` directory.

## Usage

To run the player re-identification pipeline:

```bash
python src/main.py
```
The script will process the video frame by frame, perform detections, extract features, apply re-identification logic, and save the annotated output video.

## Output

A new video file named `15sec_output_tracked.mp4` will be generated in the `output/` directory. This video will display bounding boxes, class labels (e.g., "Player", "GK", "Ball", "Ref"), and assigned tracking IDs for each detected object.

## Challenges and Future Improvements

While the core pipeline for player re-identification has been implemented, several challenges were encountered, particularly regarding the accuracy and consistency of player ID assignment:

1.  Feature Discriminability with Generic ResNet:
    
    -   Challenge: The use of a generic `ResNet50` model (pre-trained on ImageNet for general object classification) as a feature extractor proved to be a primary limitation. Its features are not inherently optimized for the fine-grained task of person re-identification in dynamic sports footage, where players can have similar appearances, varying poses, and changing lighting conditions.
        
    -   Outcome: Despite extensive tuning of the `REID_THRESHOLD` (e.g., from `0.5` to `100.0`), achieving truly consistent and unique IDs for individual players throughout the entire video remained a challenge. This was evident from the high number of unique IDs assigned (e.g., ~3000-5000 IDs for roughly 22 players over 15 seconds), indicating frequent re-assignment of new IDs to the same player.
        
2.  Threshold Sensitivity:
    
    -   Challenge: Finding an optimal fixed `REID_THRESHOLD` for L2 distance from generic features was difficult. A low threshold led to new IDs for every slight variation (false negatives), while a high threshold risked assigning the same ID to different players (false positives).
        

### Future Improvements / How to Proceed with More Time/Resources:

To significantly enhance the accuracy and reliability of player re-identification, I would propose the following:

1.  Dedicated Person Re-ID Models:
    
    -   Replace the generic `ResNet50` feature extractor with models specifically designed and trained for person re-identification. Models like OSNet, PCB (Part-based Convolutional Baseline), MGN (Multiple Granularity Network), or recent Transformer-based Re-ID models (e.g., TransReID) are often trained with metric learning losses (like Triplet Loss or Contrastive Loss). This training forces them to learn highly discriminative features, making features of the same person very similar and features of different persons distinctly different.
        
2.  Advanced Multi-Object Tracking Frameworks:
    
    -   Integrate a robust multi-object tracking framework such as DeepSORT (Deep Simple Online and Realtime Tracking) or ByteTrack. These systems go beyond simple feature matching by combining:
        
        -   Appearance Embeddings: Utilizing features from a dedicated Re-ID model.
            
        -   Motion Models: Employing techniques like Kalman Filters to predict object positions and handle short-term occlusions based on motion.
            
        -   Association Algorithms: Using methods like the Hungarian algorithm to associate new detections with existing tracks based on both appearance and motion cues, providing much more stable and long-term track IDs.
            
3.  Temporal Consistency and Smoothing:
    
    -   Implement additional post-processing steps to ensure temporal consistency of tracks, potentially using interpolation or smoothing algorithms to bridge small gaps in detection or re-identification.
        
4.  Ensemble or Fusion Approaches:
    
    -   Explore combining multiple cues (e.g., appearance, motion, contextual information like team color) to improve robustness, especially in challenging scenarios.
