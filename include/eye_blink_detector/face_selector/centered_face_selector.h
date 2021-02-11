#ifndef EYE_BLINK_DETECTOR_CENTERED_FACE_SELECTOR_H
#define EYE_BLINK_DETECTOR_CENTERED_FACE_SELECTOR_H

#include "eye_blink_detector/face_selector/face_selector.h"

namespace eb_detector {
    class CenteredFaceSelector : public FaceSelector {
    public:
        CenteredFaceSelector(const YAML::Node &params1, const YAML::Node &params);

        CenteredFaceSelector(const YAML::Node &params);

        virtual bool selectFace(const std::vector<Face>& pResults, const cv::Mat& image) override;

        float distance(const cv::Point& point_a, const cv::Point& point_b);
    protected:
        float face_conf_th_;    // Confidence threshold below which a face is not considerated
    };
}

#endif //EYE_BLINK_DETECTOR_CENTERED_FACE_SELECTOR_H
