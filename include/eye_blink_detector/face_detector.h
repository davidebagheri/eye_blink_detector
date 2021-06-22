#ifndef EYE_BLINK_DETECTOR_FACE_DETECTOR_H
#define EYE_BLINK_DETECTOR_FACE_DETECTOR_H

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "facedetectcnn.h"
#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/face_selector/centered_face_selector.h"
#include "eye_blink_detector/face_selector/biggest_face_selector.h"
#include "eye_blink_detector/face.h"
#include "eye_blink_detector/utils.h"

//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

namespace eb_detector {
    class FaceDetector {
    public:
        FaceDetector(const YAML::Node& params);

        void detectFaces(const cv::Mat& image);

        bool selectFace(const cv::Mat& image);

        const Face& getSelectedFace();

        void visualizeResults(cv::Mat* image);

    private:
        void printFace(cv::Mat* image, const Face& face, const cv::Scalar& color);

        void printResults();

        void convertResults(int* pResults);

        // Face detection buffers
        unsigned char * pBuffer_;           // Buffer used in the detection functions.
        std::vector<Face> detected_faces_;

        // Face selector
        FaceSelector* face_selector_;

        // Params
        int pyr_down_;
        std::string face_selector_type_;
        bool verbose_;
    };
}

#endif //EYE_BLINK_DETECTOR_FACE_DETECTOR_H
