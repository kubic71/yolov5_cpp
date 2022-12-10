#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"

using namespace std;


string segm_predictions_to_str(at::Tensor detections, vector<vector<cv::Point>> segments) {
    stringstream ss;

    for (int i = 0; i < detections.size(0); ++i) {
        auto x1 = detections[i][0].item<float>();
        auto y1 = detections[i][1].item<float>();
        auto x2 = detections[i][2].item<float>();
        auto y2 = detections[i][3].item<float>();

        ss << detections[i][5].item<int>() << " " << detections[i][4].item<float>() << " ";
        ss << x1 << " " << y1 << " " << x2 << " " << y2 << " ";

        auto pts = segments[i];
        for (auto pt : pts) {
            ss << pt.x << " " << pt.y << " ";
        }
        ss << endl;
    }

    return ss.str();
}


tuple<at::Tensor, at::Tensor, vector<vector<cv::Point>>> segm_predict(cv::Mat cv_img, torch::jit::script::Module& model, float conf_thresh, float iou_thresh, int img_size) {
    at::Tensor img = preprocess_image(cv_img, img_size);

    auto output = model.forward({img}).toTuple();

    at::Tensor pred = output->elements()[0].toTensor();
    at::Tensor proto = output->elements()[1].toTensor();

    // we have batch size of 1, so we can remove the first dimension
    pred = pred.squeeze(0);
    proto = proto.squeeze(0);

    auto detections = process_detections(pred, conf_thresh, 32);

    std::vector<int> keep = non_max_suppression(detections, conf_thresh, iou_thresh);

    detections = detections.index_select(0, torch::tensor(keep).to(torch::kCUDA));

    auto masks = process_masks(proto, detections, img.size(3), img.size(2));
    cout << "Masks shape: " << masks.sizes() << endl;

    auto segments = masks_to_segments(masks);

    scale_segments(segments, img.size(3), img.size(2), cv_img.cols, cv_img.rows);
    scale_boxes(detections, img.size(3), img.size(2), cv_img.cols, cv_img.rows);
    masks = scale_masks(masks, img.size(3), img.size(2), cv_img.cols, cv_img.rows);


    return {detections.slice(1, 0, 6), masks, segments};
}
