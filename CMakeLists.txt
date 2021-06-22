cmake_minimum_required(VERSION 3.10)
project(eye_blink_detector)

# Find packages
find_package( OpenCV REQUIRED )
find_package( facedetection REQUIRED )
find_package(ngraph REQUIRED)

# Set libfacedetection paths
set(facedetection_INCLUDE_DIRS "/usr/local/include/facedetection")
set(facedetection_LIBS "facedetection")

# Set yaml-cpp paths
set(yaml_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/yaml-cpp/include")
set(yaml_LIBS "${CMAKE_SOURCE_DIR}/yaml-cpp/build/libyaml-cpp.so")

# C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include dirs
include_directories(
		${facedetection_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
        ${NGraph_INCLUDE_DIRS}
		${yaml_INCLUDE_DIRS}
		include
)

# Library
add_library(${PROJECT_NAME} SHARED
		#src/utils.cpp
		src/face_detector.cpp
		src/eye_blink_detector.cpp
		src/blink_classifier.cpp
		# Face selectors
		src/face_selector/centered_face_selector.cpp
		src/face_selector/biggest_face_selector.cpp
		# Eye croppers
		src/eyes_cropper/eyes_band_cropper.cpp
		src/eyes_cropper/separate_eyes_cropper.cpp
		# Feature extractors
		src/feature_extractor/nn_feature_extractor.cpp
		src/feature_extractor/segmentation_extractor.cpp
		# Classifiers
		src/classifier/sequence_classifier.cpp
		src/classifier/nn_classifier.cpp
		)

target_link_libraries(${PROJECT_NAME}
		${OpenCV_LIBS}
		${facedetection_LIBS}
		${yaml_LIBS}
		)

# Eye Blink detection from a streaming camera
add_executable(main main.cpp)
target_link_libraries(main
		${PROJECT_NAME}
		${yaml_LIBS}
		)