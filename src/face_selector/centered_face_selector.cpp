#include "eye_blink_detector/face_selector/centered_face_selector.h"
#include <math.h>

namespace eb_detector{
    CenteredFaceSelector::CenteredFaceSelector(const YAML::Node &params)
            : FaceSelector(params) {
        // Params
        getParam(params, "confidence_threshold", this->face_conf_th_, 95.0f);
    }
    bool CenteredFaceSelector::selectFace(const std::vector<Face>& detected_faces, const cv::Mat& image) {
        cv::Point img_center = cv::Point(image.cols/2, image.rows/2);
        float distance_to_center = INFINITY;
        for (const auto& face : detected_faces) {
            // Threshold on the detection confidence
            if (face.confidence > face_conf_th_){
                cv::Point face_center = cv::Point(face.bounding_box.x + face.bounding_box.width/2,
                        face.bounding_box.y + face.bounding_box.height/2);
                float current_dist_from_center = distance(face_center, img_center);

                // Take the most centered face
                if (current_dist_from_center < distance_to_center){
                    selected_face_ = face;
                    distance_to_center = current_dist_from_center;
                }
            }
        }
        if (distance_to_center == INFINITY) return false;   // No face selected

        return true;
    }

    float CenteredFaceSelector::distance(const cv::Point& point_a, const cv::Point& point_b){
        return std::sqrt(std::pow(point_a.x-point_b.x, 2) + std::pow(point_a.y-point_b.y, 2));
    }
}