# Brief Report: Player Re-Identification in Sports Footage

## 1. Introduction
This report details the implementation of a player re-identification system for sports footage, developed as part of the AI Intern assignment for Liat.ai. The objective was to identify and assign consistent unique IDs to players within a 15-second football video, even if they temporarily leave and re-enter the frame.

## 2. Approach and Methodology

The solution is structured as a multi-stage pipeline, leveraging computer vision and machine learning techniques:

### 2.1. Object Detection
* **Model:** The provided pre-trained **Ultralytics YOLOv11** model, fine-tuned for sports analytics, was used. This model is capable of detecting and classifying `ball`, `goalkeeper`, `player`, and `referee` entities within each frame.
* **Purpose:** To generate bounding box detections and class labels for all relevant objects in the video, serving as the input for subsequent stages.

### 2.2. Feature Extraction
* **Model:** A **ResNet50** Convolutional Neural Network, pre-trained on the ImageNet dataset, was employed as the feature extractor. The final classification layer was removed, allowing the model to output 2048-dimensional feature vectors (embeddings) for each detected object.
* **Purpose:** To generate a unique, high-dimensional numerical representation (feature vector) for each detected player and goalkeeper. These features are intended to capture the visual characteristics necessary for distinguishing individuals. The model runs on CPU, with logic for GPU acceleration if available.

### 2.3. Similarity Search and Re-identification Logic
* **Database:** A **FAISS (Facebook AI Similarity Search) IndexFlatL2** index was used. This index efficiently stores the extracted feature vectors and enables rapid nearest-neighbor searches based on L2 (Euclidean) distance.
* **ID Assignment Strategy:**
    * **Ball Tracking:** The main `ball` (Class ID 0) is assigned a fixed ID (`-1`) for consistent tracking throughout the video, based on the highest confidence detection in each frame.
    * **Referee Tracking:** Referees (Class ID 3) are assigned temporary, unique string-based IDs (`Ref_X`) per detection, as persistent re-identification was not the primary focus for them in this assignment's scope.
    * **Player/Goalkeeper Re-identification (Classes 1 and 2):**
        * For each new detection of a player or goalkeeper, its extracted feature vector is queried against the `FAISS` index containing features of previously identified players.
        * If a nearest neighbor is found within a predefined `REID_THRESHOLD` (L2 distance), the existing ID associated with that neighbor is assigned to the current detection. This indicates a successful re-identification.
        * If no sufficiently close match is found, a new unique ID is generated and assigned to the current detection. Its feature vector is then added to the `FAISS` index along with its new ID, expanding the database of known players.
        * The `REID_THRESHOLD` was set to `100.0` after iterative tuning.

### 2.4. Output Generation
* **Visualization:** Bounding boxes, class labels, confidence scores, and assigned track IDs are drawn directly onto each video frame.
* **Video Export:** The annotated frames are compiled and saved as a new `.mp4` video file (`15sec_output_tracked.mp4`) in the `output/` directory, allowing for visual inspection of the tracking performance.

## 3. Techniques Tried and Their Outcomes

* **Initial Setup & Detection (Stage 1):** Successfully loaded the video and integrated the YOLOv11 model to detect objects. This stage worked as expected, providing bounding boxes for all entities.
* **Feature Extraction & FAISS Integration (Stage 2):** Successfully extracted 2048-dimensional features using ResNet50 and integrated these into a FAISS `IndexFlatL2` for similarity search.
* **Re-identification with Thresholding (Stage 3 - Iterative Tuning):**
    * **Initial Threshold (0.5):** An initial `REID_THRESHOLD` of `0.5` (L2 distance) was tested. This proved to be excessively strict. The system assigned a new unique ID to almost every detection in every frame, resulting in thousands of unique IDs (e.g., `5460` over 15 seconds) and minimal re-identification. This indicated that features of the same player across frames, even with minor variations, were yielding distances greater than `0.5`.
    * **Increased Thresholds (50.0, 70.0, 100.0):** The threshold was progressively increased to `50.0`, then `70.0`, and finally `100.0`. This significantly reduced the total number of unique IDs generated (e.g., `3261` with `REID_THRESHOLD = 100.0`). While this improved the re-identification rate, perfect consistency was still not achieved for all players throughout the entire video.
    * **Ball Tracking Outcome:** The logic for tracking the main ball with a consistent ID (`-1` / "Main") was successful and robust throughout the video.

## 4. Challenges Encountered

Several technical and algorithmic challenges were faced during the development:

* **Python Module Import Issues (`ModuleNotFoundError`):** Initial attempts to run the script directly from a sub-directory led to import errors. This was resolved by explicitly adding the project root to `sys.path`, ensuring Python correctly recognized local modules.
* **OpenCV GUI Backend Error (`cv2.error`):** An issue with `cv2.imshow` (and potentially video writing) related to missing GUI backend support in the default `opencv-python` installation on Windows was encountered. This was resolved by uninstalling `opencv-python` and installing `opencv-contrib-python`, which includes the necessary GUI components.
* **Mixed Data Types for `track_id` (`ValueError`):** The `draw_bboxes` utility function initially attempted to cast all `track_id` values to integers. However, temporary IDs for referees and extra balls were string-based (e.g., "Ref\_14"). This caused a `ValueError`. The fix involved enhancing the `draw_bboxes` function to intelligently handle both integer and string-based IDs.
* **Core Algorithmic Challenge: Discriminative Power of Generic Features for Re-ID:** This was the most significant challenge. `ResNet50` (trained on ImageNet) provides powerful generic features, but they are not inherently optimized for the nuanced task of distinguishing individual people in varying poses, lighting conditions, and partial occlusions in a dynamic environment like sports footage. The L2 distance in the 2048-dimensional feature space often proved ambiguous; a threshold sufficient to match a single player across frames might also be too broad, potentially leading to false positives, or too strict, leading to new IDs for the same player. This fundamental limitation of the feature extractor affected the overall accuracy and reliability of player re-identification.

## 5. What Remains and How I Would Proceed with More Time/Resources

The current solution demonstrates a functional pipeline for player re-identification, but the primary limitation is the consistency of player IDs due to the feature extractor's generic nature. Given more time and resources, I would focus on the following key improvements:

1.  **Utilize Dedicated Person Re-Identification Models:**
    * Replace `ResNet50` with state-of-the-art models explicitly designed and trained for person re-identification. Examples include **OSNet, PCB (Part-based Convolutional Baseline), MGN (Multiple Granularity Network), or recent Transformer-based Re-ID architectures (e.g., TransReID)**. These models are typically trained using metric learning losses (e.g., Triplet Loss, Contrastive Loss) on large-scale Re-ID datasets (like Market-1501, DukeMTMC-reID), which enables them to learn highly discriminative features that are robust to viewpoint, pose, and illumination changes.

2.  **Integrate Advanced Multi-Object Tracking Frameworks:**
    * Implement or integrate robust multi-object tracking (MOT) frameworks like **DeepSORT (Deep Simple Online and Realtime Tracking) or ByteTrack**. These frameworks combine appearance information (from dedicated Re-ID embeddings) with motion cues (e.g., using Kalman Filters for state prediction) to build and maintain long-term, consistent tracks. They are specifically designed to handle challenges like occlusions and re-entry into the frame more effectively.

3.  **Refine Similarity Metric and Thresholding:**
    * Beyond L2 distance, explore other similarity metrics (e.g., Cosine Similarity). With dedicated Re-ID features, the thresholding process would become more robust and less sensitive.
    * Investigate adaptive or dynamic thresholding methods, perhaps based on the confidence of the detection or the stability of the track.

4.  **Incorporate Temporal Smoothing and Track Management:**
    * Implement additional logic to manage track lifecycles, such as pruning stale tracks (those not seen for a certain number of frames) from `active_tracked_ids` and potentially from the FAISS index (if using a FAISS index type that supports removal efficiently, like `IndexIVFFlat`).
    * Introduce mechanisms for track interpolation to bridge short periods of missed detections.

These advancements would significantly improve the accuracy, reliability, and real-world applicability of the player re-identification system.

---
