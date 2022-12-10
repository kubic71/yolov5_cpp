#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"

using namespace std;


const float CONF_THRESH = 0.1;
const float IOU_THRESH = 0.45;

void save_results(at::Tensor detections, string filename) {

    ofstream f(filename);

    // each line contains class, confidence, x1, y1, x2, y2
    for (int i = 0; i < detections.size(0); ++i) {
        auto x1 = detections[i][0].item<float>();
        auto y1 = detections[i][1].item<float>();
        auto x2 = detections[i][2].item<float>();
        auto y2 = detections[i][3].item<float>();

        f << detections[i][5].item<int>() << " " << detections[i][4].item<float>() << " ";
        f << x1 << " " << y1 << " " << x2 << " " << y2;

        f << endl;
    }

    f.close();
}


at::Tensor predict(cv::Mat cv_img, torch::jit::script::Module& model, float conf_thresh, float iou_thresh) {
    at::Tensor img = preprocess_image(cv_img);

    at::Tensor output = model.forward({img}).toTuple()->elements()[0].toTensor();

    // we have batch size of 1, so we can remove the first dimension
    output = output.squeeze(0);
    auto detections = process_detections(output, conf_thresh);


    std::vector<int> keep = non_max_suppression(detections, conf_thresh, iou_thresh);

    detections = detections.index_select(0, torch::tensor(keep).to(torch::kCUDA));

    scale_boxes(detections, img.sizes()[3], img.sizes()[2], cv_img.cols, cv_img.rows);

    return detections;
}



int main() {

    // Load the YOLOv5 network from a PyTorch checkpoint file
    auto model = load_model("yolov5s.torchscript");

    // Load the input image "zidane.jpg"
    cv::Mat cv_img = cv::imread("zidane.jpg");
    // convert BGR format to RGB
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);

    at::Tensor detections = predict(cv_img, model, CONF_THRESH, IOU_THRESH);
        // get the number of boxes
    cout << "detections: " << detections << endl;

    draw_bounding_boxes(detections, cv_img);

    // save the image
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
    cv::imwrite("output.jpg", cv_img);

    save_results(detections, "output.txt");


    return 0;
}