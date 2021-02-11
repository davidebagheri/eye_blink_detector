#ifndef EYE_BLINK_DETECTOR_FACE_SELECTOR_H
#define EYE_BLINK_DETECTOR_FACE_SELECTOR_H

#include <opencv2/core/types.hpp>

#include "yaml-cpp/yaml.h"
#include "eye_blink_detector/face.h"
#include "eye_blink_detector/utils.h"

namespace eb_detector {

    class FaceSelector {
    public:
        FaceSelector(const YAML::Node& params){};

        virtual bool selectFace(const std::vector<Face>& detected_faces, const cv::Mat& frame) = 0;

        const Face& getSelectedFace(){
            return selected_face_;
        }

    protected:
        Face selected_face_;
    };
}

#endif //EYE_BLINK_DETECTOR_FACE_SELECTOR_H
