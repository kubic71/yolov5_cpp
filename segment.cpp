#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"
#include "segment.hpp"

using namespace std;


const float CONF_THRESH = 0.2;
const float IOU_THRESH = 0.45;
const int IMG_SIZE = 640;




void save_results(at::Tensor detections, vector<vector<cv::Point>> segments, string filename) {
    ofstream f(filename);
    f << segm_predictions_to_str(detections, segments);
    f.close();
}







int main() {

    // Load the YOLOv5 network from a PyTorch checkpoint file
    auto model = load_model("yolov5s-seg.torchscript");

    // Load the input image "zidane.jpg"
    cv::Mat cv_img = cv::imread("bus.jpg");
    // convert BGR format to RGB
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);

    auto tmp = segm_predict(cv_img, model, CONF_THRESH, IOU_THRESH, IMG_SIZE);
    auto detections = get<0>(tmp);
    auto masks = get<1>(tmp);
    auto segments = get<2>(tmp);

        // get the number of boxes
    cout << "detections: " << detections << endl;

    draw_masks(detections, masks, cv_img);
    draw_bounding_boxes(detections, cv_img);

    // save the image
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
    cv::imwrite("output.jpg", cv_img);

    save_results(detections, segments, "output.txt");

    // cv::imshow("image", cv_img);
    // cv::waitKey(0);


    return 0;
}