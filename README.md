# Project 3: Real-time 2-D Object Recognition

**CS5330 Pattern Recognition & Computer Vision**

## Team

- GitHub: Parker Cai - [@parkercai (https://github.com/ParkerCai)]
- GitHub: Jenny Nguyen - [@jennyncodes (https://github.com/jennyncodes)]

## Overview

Build a real-time 2-D object recognition system that identifies objects on a uniform surface in a translation, scale, and rotation invariant manner.

## Project Description

Given a live video feed or a set of images, the system:

1. Thresholds the input to separate objects from a white background
2. Cleans the binary image using morphological filtering
3. Segments individual regions via connected components analysis
4. Computes rotation/scale-invariant features (axis of least central moment, oriented bounding box, Hu moments)
5. Train new objects in the interactive training mode
6. Classifies each object against a trained database using nearest-neighbor matching with scaled Euclidean distance

An embedding-based classification pipeline using a DNN (ResNet18) is also explored for comparison.

## Directory Structure

```
Project_3_Real-time_2-D_Object_Recognition/
├── bin/                    # Executables (auto-generated)
├── build/                  # CMake build files (auto-generated)
├── release/                # Distribution packages (auto-generated)
├── data/
├── include/                # Header files
├── src/                    # Source files
│   ├── CMakeLists.txt      # Build configuration
│   ├── or2d.cpp            # Main program
│   └── gui/                # Dear ImGui GUI application
│       ├── cbir_gui.cpp    # GUI main source
│       ├── app_icon.ico    # Windows executable icon
│       └── app_icon.rc     # Windows resource file
├── report/                 # Report documents
├── build.bat               # Build script
├── package_release.bat     # Release packaging script
└── README.md
```

## Build Instructions

### Option 1: VS Code with CMake Tools

1. Open the `src/` folder in VS Code
2. CMake Tools will auto-detect `CMakePresets.json`
3. Click **Build** button in the status bar at the bottom

### Option 2: Command Line

```bash
cd src
cmake --preset default
cmake --build --preset release
```

### Option 3: Manual (Most reliable, just run the ./build.bat file at root)

```bash
cd Assignment1/src
mkdir build
cd build
cmake -Wno-dev -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv_build\build\install" ../src
cmake --build . --config Release
```

<!-- How to run the program -->

## Running

```bash
# default camera
.\bin\or2d.exe

# or specify camera number
.\bin\or2d.exe 1
```

### Controls

- `q` - quit
- `s` - save images (for report)
- `a` - toggle auto/manual threshold
- `t` - toggle training mode
- `n` - Save training example (in training mode)
- `e` - Toggle evaluation mode
- `f` - Toggle cnn mode
- `c` - Record cnn results
- `r` - Record evaluation result (in eval mode)
- `p` - Print confusion matrix
- `m` - Print cnn confusion matrix
- `u` - Toggle unknown detection (extension)
- `l` - Learn unknown object (extension)
- `+`/`-` - adjust threshold
- `1` - Show original
- `2` - Show threshold only
- `3` - Show cleaned (morphology)
- `4` - Show features (OBB + axis)
- `5` - Show classification
- `6` - Show CNN
- `h` - help

## Tasks

### Task 1: Thresholding

- **Implementation**: ISODATA algorithm (k-means with k=2) for authomatic threshold calculation built from scratch
- **Method**: Sample 1/16 of pixels, run k-means to find object and background clusters, use midpoint as threshold
- **From Scratch**: Manual pixel-by-pixel thresholding loop (NOT using cv::threshold())
- **File**: `src/thresholding.cpp`
- **Testing**: .\bin\or2d.exe and press `1` to view thresholded output

### Task 2: Morphological Filtering (Clean Up)

- **Implementation**: Erosion and dilation operations built from scratch
- **Strategy**:
  - Opening (erode → dilate) to remove noise
  - Closing (dilate → erode) to fill holes
- **From Scratch**: Manual neighbor-checking loops for erosion and dilation (NOT using cv::erode() or cv::dilate())
- **File**: `src/morphology.cpp`
- **Testing**: Run program and press `2` to view cleaned output

### Task 3: Connected Components (Segmentation)

- **Implementation**: Region segmentation using OpenCV's `connectedComponentsWithStats` (Spaghetti4C / Bolelli et al. 2021 two-pass union-find with DAG-based decision trees)
- **Filtering**:
  - Ignores regions smaller than a minimum area (default 400 px = 20×20)
  - Skips regions touching the image border
  - Keeps only the top N largest regions (default N=3) for multi-object recognition
- **Display**: Color-coded region map using hardcoded color palette. Centroid matching between frames prevents color flickering
- **Overlays**: Axis-aligned bounding boxes (AABB) and centroids (white dots) drawn on the color-coded result
- **File**: `src/segmentation.cpp`
- **Testing**: Run program and press `3` to view segmented regions

### Task 4: Feature Computation

- **Implementation**: Region-based analysis (not boundary-based) for translation, scale, and rotation invariant features
- **Computed features**:
  1. **Axis of least central moment** (theta): `0.5 * atan2(2 * mu11, mu20 - mu02)` — the angle that minimizes the moment of inertia through the centroid
  2. **Oriented bounding box (OBB)**: Projects all region pixels onto the principal and perpendicular axes relative to the centroid, tracking min/max extents to get the OBB
  3. **Percent filled**: Region area / OBB area — measures how much of the OBB is occupied (invariant to translation, scale, rotation)
  4. **OBB aspect ratio**: min(width, height) / max(width, height), always in [0, 1] (invariant to translation, scale, rotation)
  5. **Hu moment invariants**: Computed via `cv::HuMoments()` from `cv::moments()` — 7 moments invariant to translation, scale, and rotation. Uses hu[0] and hu[1] as log10(|hu[i]|) for manageable magnitude
- **Feature vector**: {percentFilled, bboxRatio, log|hu0|, log|hu1|} (4-d)
- **Moments used**: `cv::moments(mask, true)` computes spatial moments (m00, m10, m01, ...), central moments (mu20, mu11, mu02, ...), and normalized central moments. `binaryImage=true` treats any nonzero pixel as 1
- **Display**: OBB drawn as a white polygon, principal axis as a blue line, centroid as a red dot, percent filled and aspect ratio as text
- **File**: `src/features.cpp`
- **Testing**: Run program and press `4` to view features (OBB + axis) overlaid on the color-coded regions

### Task 5: Training Data Collection

- **Implementation**: Interactive training mode
- **How it works**:
  - Position object in view
  - Enter object name when prompted
  - Repeat 3-5 times per object at different orientations
- **Database**: saves to data/objects_db.csv
- **File**: `src/training.cpp`
- **Testing**: Run program and press `4` to view features and then press `t` to enter training mode, and press `n` to save

### Task 6: Classification (Hand-built features)

- **Implementation**: Nearest-neighbor with scaled Euclidean distance using sqrt(Σ((f1[i] - f2[i]) / stddev[i])²)
- **Features**: Normalizes by standard deviation for equal weighting
- **Confidence**: Calculated as 1 / (1 + distance)
- **File**: `src/classification.cpp`
- **Database**: Saves (4-d) hand-built features to `data/objects_db.csv`
- **Testing**: Run program and press `5` to view classification with labels

### Task 7: Evaluation (Confusion Matrix)

- **Implementation**:  Interactive evaluation mode
- **Features**: Normalizes by standard deviation for equal weighting
- **Confidence**: Calculated as 1 / (1 + distance)
- **File**: `src/evaluation.cpp`
- **Testing**: Run program and press `e` to enter evaluation mode, then press `r` to record the object and press `p` to print confusion matrix

### Task 8: Demo Video

### Task 9: Embedding-based Classification

- **Implementation**: One-shot classification using CNN (ResNet18) image embeddings
- **Pipeline**:
  1. Uses centroid (cx, cy), principal axis angle (theta), and OBB extents (uMin, uMax, vMin, vMax) from the feature computation step
  2. Rotates the original image so the region's primary axis is aligned with the X-axis (rotate -theta)
  3. Extracts an axis-aligned region of interest (ROI) from the rotated image corresponding to the region's bounding box
  4. Reshapes the ROI to 224×224 for CNN input
  5. Passes the image through a pre-trained ResNet18 network to get a 512-d embedding from the second to last layer
- **Model**: ResNet18 ONNX model (`data/CNN/resnet18-v2-7.onnx`)
- **Distance Metric**: Sum-squared difference (SSD) between embedding vectors
- **One-shot**: Only requires a single training example per object class
- **Database**: Saves (512-d) CNN embeddings to `data/objects_cnn_db.csv`
- **Utilities**: `prepEmbeddingImage()` and `getEmbedding()` in `src/utilities.cpp` (based on code by Prof. Bruce A. Maxwell)
- **Files**: `src/utilities.cpp`, `src/classification.cpp`, `src/or2d.cpp`
- **Testing**: Run program, press `t` for training mode, press `c` to save a CNN embedding, then press `6` to view CNN classification. Press `5` to compare against hand-built feature classification.

## Extensions
### Unknown Object Detection & Auto Learning

- **Implementation**:  Automatic detection of unknown objects based on confidence threshold
- **Features**: System monitors classification confidence and objects with confidence < 50% flagged as "Unknown" will be in red. System will prompt user for the name if user wants to learn the new object and add to database. Object will turn yellow once it is known.
- **File**: `src/unknown.cpp`
- **Testing**: Run program and press `u` to enter unknown detection mode, then press `5` to show classification view and press `l` to learn new object 

### Extended Object Database 
- **Implementation**: We entered 5 more objects in the Interactive training mode
- **Database**: saves to data/objects_db.csv
- **File**: `src/training.cpp`
- **Testing**: Run program and press `4` to view features and then press `t` to enter training mode, and press `n` to save

## Demo

<!-- Link to demo video -->
