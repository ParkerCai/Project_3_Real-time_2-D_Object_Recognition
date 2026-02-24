cmake_minimum_required(VERSION 3.24)
project(or2d)  # Real-time 2-D Object Recognition

# Set C++ standard to C++23 for modern features like println
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Fetch GLFW
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.1
)
FetchContent_MakeAvailable(imgui)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../include)      # Your headers
include_directories(${CMAKE_SOURCE_DIR}/csv_util)        # csv_util
include_directories(${imgui_SOURCE_DIR})                 # ImGui headers
include_directories(${imgui_SOURCE_DIR}/backends)        # ImGui backend headers

# Output executables to bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/../bin)

# --- Core source files ---
set(SOURCES
    threshold.cpp
    morphology.cpp
    segmentation.cpp
    features.cpp
    training.cpp
    classification.cpp
    evaluation.cpp
    utilities.cpp
    unknown.cpp
)

# --- ImGui source files (using OpenGL2 backend - simpler, no loader needed) ---
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
)



# --- Executables ---

# OR2D main program (CLI)
add_executable(or2d or2d.cpp ${SOURCES})
target_link_libraries(or2d ${OpenCV_LIBS})

# OR2D GUI program (WIN32 hides console window)
# add_executable(or2d_gui WIN32 gui/or2d_gui.cpp gui/app_icon.rc ${SOURCES} ${IMGUI_SOURCES})
# target_link_libraries(or2d_gui ${OpenCV_LIBS} glfw OpenGL::GL dwmapi)
# set_target_properties(or2d_gui PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
