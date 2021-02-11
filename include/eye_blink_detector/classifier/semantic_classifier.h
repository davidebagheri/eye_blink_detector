#ifndef EYE_BLINK_DETECTOR_SEMANTIC_CLASSIFIER_H
#define EYE_BLINK_DETECTOR_SEMANTIC_CLASSIFIER_H

#include "eye_blink_detector/classifier/nn_classifier.h"

namespace eb_detector {
    class SemanticClassifier : public NNClassifier{
    public:
        SemanticClassifier(const YAML::Node& params);

        // Check if in the sequence there is only a single sub-sequence of high value ( > seg_threshold_) samples
        bool hasSinglePeak(std::vector<std::vector<int>>* peaks);

        // Check if the peaks overlap in time
        bool arePeaksOverlapping(const std::vector<std::vector<int>>& peaks);

        bool arePeaksOverlapping(const std::vector<int> peak_a, const std::vector<int>& peaks_b);

        virtual bool forward(const cv::Mat& features) override;
    private:
        /* Classification params
         * A blink is considered valid only of the reference_class_ has a value above seg_threshold_
         */
        float seg_threshold_;
        int reference_class_;
    };
}

#endif //EYE_BLINK_DETECTOR_SEMANTIC_CLASSIFIER_H
