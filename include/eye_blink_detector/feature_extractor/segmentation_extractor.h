#ifndef EYE_BLINK_DETECTOR_SEGMENTATION_EXTRACTOR_H
#define EYE_BLINK_DETECTOR_SEGMENTATION_EXTRACTOR_H

#include "eye_blink_detector/feature_extractor/nn_feature_extractor.h"

namespace eb_detector {
    class SegmentationExtractor : public NNFeatureExtractor{
    public:
        SegmentationExtractor(const YAML::Node& params);

        virtual cv::Mat forward(const std::vector<cv::Mat>& model_in) override;

        virtual void visualizeResults(cv::Mat* image) override;

        cv::Mat computeArgmax(const cv::Mat& model_output);

        std::vector<float> computeClassesAreas(const cv::Mat& prediction);

        cv::Mat getColoredSegmentationMap(const cv::Mat& prediction);

    protected:
        // Parameters
        int n_classes_;
        int class_to_return_;   // if <0 all the classes densities are returned, othewise the one selected in this param
        bool visualize_;
        std::vector<cv::Vec3b> colors_;

        // Variables
        std::vector<cv::Mat> predictions_;
    };
}

#endif //EYE_BLINK_DETECTOR_SEGMENTATION_EXTRACTOR_H
