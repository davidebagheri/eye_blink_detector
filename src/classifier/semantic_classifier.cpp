#include "eye_blink_detector/classifier/semantic_classifier.h"

namespace eb_detector{
    SemanticClassifier::SemanticClassifier(const YAML::Node& params) : NNClassifier(params){
        getParam<float>(params, "seg_threshold", seg_threshold_, 0.98);
        getParam(params, "reference_class", reference_class_, 0);
    }

    bool SemanticClassifier::hasSinglePeak(std::vector<std::vector<int>>* peaks){
        for (int batch_n = 0; batch_n < batch_size_; batch_n++){
            // Control flags
            bool started = false;
            bool ended = false;

            // Single sequence result
            std::vector<int> peak_idxs; // vector containing the start and end indexes of the peak

            // the sequence can not start with a peak
            if (seq_blob_.at<float>(batch_n, 0, reference_class_) > seg_threshold_) return false;

            for (int time_step = 1; time_step < sequence_len_; time_step++){
                /*if (seq_blob_.at<float>(batch_n, time_step, reference_class_) >= seg_threshold_){
                    std::cout << "*";
                } else std::cout << "_";*/
                if (!started){
                    // first sample of the peak met
                    if (seq_blob_.at<float>(batch_n, time_step, reference_class_) >= seg_threshold_){
                        peak_idxs.push_back(time_step);
                        started = true;
                        continue;
                    }
                } else {
                    // last sample of the peak met
                    if (!ended){
                        if (seq_blob_.at<float>(batch_n, time_step, reference_class_) < seg_threshold_) {
                            peak_idxs.push_back(time_step - 1);
                            ended = true;
                            continue;
                        }
                    } else {
                        // Another peak encountered
                        if (seq_blob_.at<float>(batch_n, time_step, reference_class_) >= seg_threshold_) {
                            reset();
                            //std::cout << std::endl;
                            return false;
                        }
                    }
                }

                // No peak met
                if (time_step == sequence_len_-1 and (!started or !ended)){
                    //std::cout << std::endl;
                    return false;
                }
            }
            //std::cout << std::endl;
            peaks->push_back(peak_idxs);
        }
        return true;
    }

    bool SemanticClassifier::arePeaksOverlapping(const std::vector<int> peak_a, const std::vector<int>& peaks_b){
        if ((peak_a[0] >= peaks_b[0] and peak_a[0] <= peaks_b[1]) or
                (peak_a[1] >= peaks_b[0] and peak_a[1] <= peaks_b[1]))
            return true;
        return false;
    }

    bool SemanticClassifier::arePeaksOverlapping(const std::vector<std::vector<int>>& peaks){
        if (peaks.size() < 2) return true;

        for (int i = 0; i < peaks.size()-1; i++){
            for (int j = i+1; j < peaks.size(); j++){
                if (arePeaksOverlapping(peaks[i], peaks[j]))
                    return true;
            }
        }
        return false;
    }

    bool SemanticClassifier::forward(const cv::Mat& features){
        std::vector<std::vector<int>> peaks;

        return NNClassifier::forward(features) and hasSinglePeak(&peaks) and arePeaksOverlapping(peaks);
    }

}