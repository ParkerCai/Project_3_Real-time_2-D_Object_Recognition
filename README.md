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
5. Classifies each object against a trained database using nearest-neighbor matching with scaled Euclidean distance

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
- `+`/`-` - adjust threshold
- `1` - Show original 
- `2` - Show threshold only 
- `3` - Show cleaned (morphology) 
- `h` - help

## Tasks

### Task 1: Thresholding
- **Status**: Complete (written from scratch)
- **Implementation**: ISODATA algorithm (k-means with k=2) for authomatic threshold calculation
- **Method**: Sample 1/16 of pixels, run k-means to find object and background clusters, use midpoint as threshold
- **From Scratch**: Manual pixel-by-pixel thresholding loop (NOT using cv::threshold())
- **Testing**: .\bin\or2d.exe


### Task 2: Morphological Filtering (Clean Up)
- **Status**: Complete (written from scratch)
- **Implementation**: Erosion and dilation operations built from scratch
- **Strategy**: 
  - Opening (erode → dilate) to remove noise
  - Closing (dilate → erode) to fill holes
- **From Scratch**: Manual neighbor-checking loops for erosion and dilation (NOT using cv::erode() or cv::dilate())
- **File**: `src/morphology.cpp`
- **Testing**: Run program and press `3` to view cleaned output

### Task 3: Connected Components (Segmentation)

### Task 4: Feature Computation

### Task 5: Training Data Collection

### Task 6: Classification

### Task 7: Evaluation (Confusion Matrix)

### Task 8: Demo Video

### Task 9: Embedding-based Classification

## Extensions

## Demo

<!-- Link to demo video -->
