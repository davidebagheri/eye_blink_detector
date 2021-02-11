#ifndef EYE_BLINK_DETECTOR_BIGGEST_FACE_SELECTOR_H
#define EYE_BLINK_DETECTOR_BIGGEST_FACE_SELECTOR_H

#include "face_selector.h"

namespace eb_detector {
    class BiggestFaceSelector : public FaceSelector {
    public:
        BiggestFaceSelector(const YAML::Node &params);

        virtual bool selectFace(const std::vector<Face>& detected_faces, const cv::Mat& image) override;

    protected:
        float face_conf_th_;    // Confidence threshold below which a face is not considerated
    };
}



#endif //EYE_BLINK_DETECTOR_BIGGEST_FACE_SELECTOR_H
