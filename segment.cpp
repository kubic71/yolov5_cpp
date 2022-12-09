#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"

using namespace std;


const float CONF_THRESH = 0.2;
const float IOU_THRESH = 0.45;


// void check_model_device(torch::jit::script::Module& model) {
    // 

tuple<at::Tensor, at::Tensor> predict(cv::Mat cv_img, torch::jit::script::Module& model) {
    at::Tensor img = preprocess_image(cv_img);

    auto output = model.forward({img}).toTuple();

    at::Tensor pred = output->elements()[0].toTensor();
    at::Tensor proto = output->elements()[1].toTensor();

    // we have batch size of 1, so we can remove the first dimension
    pred = pred.squeeze(0);
    proto = proto.squeeze(0);

    auto detections = process_detections(pred, CONF_THRESH, 32);

    std::vector<int> keep = non_max_suppression(detections, CONF_THRESH, IOU_THRESH);

    detections = detections.index_select(0, torch::tensor(keep).to(torch::kCUDA));

    auto masks = process_masks(proto, detections, img.size(3), img.size(2));
    cout << "Masks shape: " << masks.sizes() << endl;

    scale_boxes(detections, img.sizes()[3], img.sizes()[2], cv_img.cols, cv_img.rows);
    masks = scale_masks(masks, img.size(3), img.size(2), cv_img.cols, cv_img.rows);
    return {detections.slice(1, 0, 6), masks};
}




int main() {

    // Load the YOLOv5 network from a PyTorch checkpoint file
    auto model = load_model("yolov5s-seg.torchscript");

    // Load the input image "zidane.jpg"
    cv::Mat cv_img = cv::imread("bus.jpg");
    // convert BGR format to RGB
    cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB);

    auto tmp = predict(cv_img, model);
    auto detections = get<0>(tmp);
    auto masks = get<1>(tmp);

        // get the number of boxes
    cout << "detections: " << detections << endl;

    draw_masks(detections, masks, cv_img);
    draw_bounding_boxes(detections, cv_img);


    // save the image
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
    cv::imwrite("output.jpg", cv_img);

    // cv::imshow("image", cv_img);
    // cv::waitKey(0);


    return 0;
}