cmake_minimum_required(VERSION 3.10)
project(eye_blink_detector)


find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

# Set libfacedetection paths
if (EXISTS "${CMAKE_SOURCE_DIR}/ThirdParty/libfacedetection/build/libfacedetection.so")
	set(facedetection_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/ThirdParty/libfacedetection/build" "${CMAKE_SOURCE_DIR}/ThirdParty/libfacedetection/src")
	set(facedetection_LIBS "${CMAKE_SOURCE_DIR}/ThirdParty/libfacedetection/build/libfacedetection.so")
else()
	message(FATAL_ERROR "You need to compile facelibdetection first.")
endif()

# Set yaml-cpp paths
if (EXISTS "${CMAKE_SOURCE_DIR}/ThirdParty/yaml-cpp/build/libyaml-cpp.so")
	set(yaml_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/ThirdParty/yaml-cpp/include")
	set(yaml_LIBS "${CMAKE_SOURCE_DIR}/ThirdParty/yaml-cpp/build/libyaml-cpp.so")
else()
	message(FATAL_ERROR "You need to compile yaml-cpp first.")
endif()


include_directories(
		${facedetection_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${yaml_INCLUDE_DIRS}
		include
)

add_library(${PROJECT_NAME} SHARED
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
target_link_libraries(main ${yaml_LIBS} ${PROJECT_NAME})
