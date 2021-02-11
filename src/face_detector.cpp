#include "eye_blink_detector/face_detector.h"

namespace eb_detector{
    FaceDetector:: FaceDetector(const YAML::Node& params){
        // params form: {face selector type, face confidence threshold, verbose}
        // Allocate buffers for face detection
        pBuffer_ = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
        if(!pBuffer_)
        {
            fprintf(stderr, "Can not alloc buffer.\n");
        }

        // Get params
        getParam(params["face_selector"], "type", face_selector_type_, std::string("biggest_face"));
        getParam(params, "verbose", verbose_, false);

        // Init face selector
        if (face_selector_type_ == "biggest_face")
            face_selector_ = new BiggestFaceSelector(params["face_selector"]);
        else if (face_selector_type_ == "centered_face")
            face_selector_ = new CenteredFaceSelector(params["face_selector"]);
        else std::cerr << "Selector type not available" <<std::endl;
    }

    void FaceDetector::detectFaces(const cv::Mat& image) {
        // Detect
        cv::TickMeter cvtm;
        cvtm.start();
        int * pResults = facedetect_cnn(pBuffer_, (unsigned char *) (image.ptr(0)),
                image.cols, image.rows, (int) image.step);
        cvtm.stop();

        // Convert result in an organized form
        convertResults(pResults);

        if (verbose_) {
            printf("time = %gms\n", cvtm.getTimeMilli());
            printf("%d faces detected.\n", (pResults ? *pResults : 0));

            printResults();
        }
    }

    void FaceDetector::convertResults(int* pResults) {
        detected_faces_.clear();

        for (int i = 0; i < (pResults ? *pResults : 0); i++) {

            short *p = ((short *) (pResults + 1)) + 142 * i;
            Face detected_face;

            detected_face.confidence = p[0];
            detected_face.bounding_box = cv::Rect(p[1], p[2], p[3], p[4]);
            detected_face.eyes[0] = cv::Point(p[5], p[5 + 1]);
            detected_face.eyes[1] = cv::Point(p[5 + 2], p[5 + 3]);

            detected_faces_.push_back(detected_face);
        }
    }

    bool FaceDetector::selectFace(const cv::Mat& image){
        return face_selector_->selectFace(detected_faces_, image);
    }

    const Face& FaceDetector::getSelectedFace(){
        return face_selector_->getSelectedFace();
    }

    void FaceDetector::visualizeResults(cv::Mat* image){
        // Visualize all the detected faces
        for (const auto& face : detected_faces_){
            char sScore[256];
            snprintf(sScore, 256, "%d", face.confidence);
            cv::putText(*image, sScore, cv::Point(face.bounding_box.x, face.bounding_box.y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,cv::Scalar(0, 255, 0), 1);

            printFace(image, face, cv::Scalar(0, 255, 0));
        }

        // Visualize the selected face
        printFace(image, face_selector_->getSelectedFace(), cv::Scalar(255, 0, 0));
    }


    void FaceDetector::printResults(){
        int face_id = 0;
        for (const auto& face : detected_faces_) {

            //print the result
            printf("face %d: confidence=%d, [%d, %d, %d, %d] (%d,%d) (%d,%d)\n",
                   face_id, face.confidence, face.bounding_box.x, face.bounding_box.y, face.bounding_box.width, face.bounding_box.height,
                   face.eyes[0].x, face.eyes[0].y, face.eyes[1].x, face.eyes[1].y);
            face_id++;
        }

    }

    void FaceDetector::printFace(cv::Mat* image, const Face& face, const cv::Scalar& color){
        //draw face rectangle
        rectangle(*image, face.bounding_box, color, 2);

        //draw five face landmarks in different colors
        cv::circle(*image, face.eyes[0], 1, color, 2);
        cv::circle(*image, face.eyes[1], 1, color, 2);
    }

}
