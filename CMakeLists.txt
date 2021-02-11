cmake_minimum_required(VERSION 3.10)
project(eye_blink_detector)

# Find packages
find_package( OpenCV REQUIRED )
find_package( facedetection REQUIRED )
find_package(ngraph REQUIRED)
find_package(yaml-cpp REQUIRED HINTS /home/davide/Desktop/eye_blink_detector/yaml-cpp)

# Set libfacedetection paths
set(facedetection_INCLUDE_DIRS "/usr/local/include/facedetection")
set(facedetection_LIBS "facedetection")

# C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include dirs
include_directories(
		${facedetection_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
        ${NGraph_INCLUDE_DIRS}
		${YAML_CPP_INCLUDE_DIR}
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
		src/classifier/semantic_classifier.cpp
)

target_link_libraries(${PROJECT_NAME}
		${OpenCV_LIBS}
		${facedetection_LIBS}
		${YAML_CPP_LIBRARIES}
		)

# Eye Blink detection from a streaming camera
add_executable(main main.cpp)
target_link_libraries(main
		${PROJECT_NAME}
		)

# Crop eyes from images in a folder
#add_executable(get_eyes_image get_eyes_image.cpp)
#target_link_libraries(get_eyes_image
#		${PROJECT_NAME}
#		)

# Crop eyes from camera video
#add_executable(get_eyes_camera get_eyes_camera.cpp)
#target_link_libraries(get_eyes_camera
#		${PROJECT_NAME}
#		)

# Crop eyes from video file
#add_executable(get_eyes_video get_eyes_video.cpp)
#target_link_libraries(get_eyes_video
#		${PROJECT_NAME}
#		)

#add_executable(dnn_prova dnn_prova.cpp)
#target_link_libraries(dnn_prova
#		${OpenCV_LIBS}
#		)