#include "eye_blink_detector/face_selector/biggest_face_selector.h"

namespace eb_detector{
    BiggestFaceSelector::BiggestFaceSelector(const YAML::Node &params)
            : FaceSelector(params) {
        // Params
        getParam(params, "confidence_threshold", this->face_conf_th_, 95.0f);
    }
    bool BiggestFaceSelector::selectFace(const std::vector<Face>& detected_faces, const cv::Mat& image) {
        float max_face_area = 0;

        for (const auto& face : detected_faces) {
            // Threshold on the detection confidence
            if (face.confidence > face_conf_th_) {
                float face_area = face.bounding_box.width * face.bounding_box.height;

                // Take the biggest face
                if (max_face_area < face_area) {
                    selected_face_ = face;
                    max_face_area = face_area;
                }
            }
        }
        if (max_face_area == 0) return false;   // No face selected

        return true;
    }
}