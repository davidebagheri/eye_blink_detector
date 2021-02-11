#include "eye_blink_detector/feature_extractor/segmentation_extractor.h"

namespace eb_detector{
    SegmentationExtractor::SegmentationExtractor(const YAML::Node& params) : NNFeatureExtractor(params){
        getParam(params, "n_classes", n_classes_, 3);
        getParam(params, "class_to_return", class_to_return_, -1);
        getParam(params, "visualize", visualize_,false);

        // Initialize some colors
        colors_ = {{255,0,0}, {0,255,0}, {0,0,255}, {255,255,0}, {0,255, 255}, {255, 0, 255}};
    }

    cv::Mat SegmentationExtractor::computeArgmax(const cv::Mat &score){
        int channels = score.size[1];
        int width = score.size[3];
        int height = score.size[2];

        cv::Mat argmax(height, width, CV_8UC1);

        // Compute argmax
        cv::Mat class_each_row (channels, width*height, CV_32FC1, score.data);
        class_each_row = class_each_row.t();
        cv::Point maxId;    // point [x,y] values for index of max
        double maxValue;    // the holy max value itself

        for (int i=0;i<class_each_row.rows;i++){
            minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);
            argmax.at<uchar>(i) = maxId.x;
        }

        return argmax;
    }

    std::vector<float> SegmentationExtractor::computeClassesAreas(const cv::Mat& prediction){
        std::vector<float> result(n_classes_, 0);

        // Count
        for (int row = 0; row < prediction.rows; row++){
            for (int col = 0; col < prediction.cols; col++){
                result[prediction.at<uchar>(row, col)]++;
            }
        }

        // Normalize
        int n_pixels = prediction.rows * prediction.cols;
        for (int i = 0; i < result.size(); i++){
            result[i] = result[i] / n_pixels;
        }

        return result;
    }

    cv::Mat SegmentationExtractor::getColoredSegmentationMap(const cv::Mat& prediction){
        cv::Mat segm(prediction.rows, prediction.cols, CV_8UC3);

        for (int row = 0; row < prediction.rows; row++){
            const uchar *ptrMaxCl = prediction.ptr<uchar>(row);
            cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);

            for (int col = 0; col < prediction.cols; col++){
                ptrSegm[col] = colors_[ptrMaxCl[col]];
            }
        }
        return segm;
    }

    cv::Mat SegmentationExtractor::forward(const std::vector<cv::Mat>& model_in) {
        // Clear previous result
        if (visualize_) {
            segmented_maps_.clear();
        }

        // Compute scores
        cv::Mat scores = NNFeatureExtractor::forward(model_in);

        // Init result
        int batch_size = scores.size[0];
        cv::Mat classes_pct;
        if (class_to_return_ < 0)
            classes_pct.create(std::vector<int>({batch_size, 1, n_classes_}), CV_32F);
        else classes_pct.create(std::vector<int>({batch_size, 1, 1}), CV_32F);

        // Process scores
        for (int i=0; i < batch_size; i++) {
            cv::Range batch_ranges[4];
            batch_ranges[0] = cv::Range(i, i+1);
            batch_ranges[1] = cv::Range::all();
            batch_ranges[2] = cv::Range::all();
            batch_ranges[3] = cv::Range::all();

            cv::Mat batch_score = scores(batch_ranges);

            // Argmax
            cv::Mat prediction = computeArgmax(batch_score);

            // Compute class areas pct
            std::vector<float> pcts = computeClassesAreas(prediction);

            // Store result
            if (class_to_return_ < 0) {
                for (int class_id = 0; class_id < pcts.size(); class_id++) {
                    classes_pct.at<float>(i, class_id) = pcts[class_id];
                }
            } else {
                classes_pct.at<float>(i, 0) = pcts[class_to_return_];
            }

            // Visualize
            if (visualize_){
                segmented_maps_.push_back(getColoredSegmentationMap(prediction));
                cv::imshow("eye " + std::to_string(i), segmented_maps_[i]);
            }
        }

        return classes_pct;
    }

}