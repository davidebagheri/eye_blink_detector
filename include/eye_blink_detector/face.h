#ifndef EYE_BLINK_DETECTOR_FACE_H
#define EYE_BLINK_DETECTOR_FACE_H

#include <opencv2/opencv.hpp>

namespace eb_detector{
    struct Face{
        cv::Rect bounding_box;  // face bounding box
        cv::Point eyes[2];      // points centered in the eyes
        int confidence;         // detection confidence
    };
}

#endif //EYE_BLINK_DETECTOR_FACE_H

