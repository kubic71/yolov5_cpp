#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>



using namespace std;

namespace F = torch::nn::functional;

cv::Mat scale_and_pad_image(cv::Mat img, int imgsz=640, int stride=32) {
    int width = img.cols;
    int height = img.rows;
    double ratio = ((double)imgsz) / std::max(width, height);

    int new_width = static_cast<int>(width * ratio);
    int new_height = static_cast<int>(height * ratio);

    float pad_w = (imgsz - new_width) % stride;
    float pad_h = (imgsz - new_height) % stride;
    pad_w /= 2;
    pad_h /= 2;

    cv::resize(img, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    int top = cvRound(pad_h - 0.1);
    int bottom = cvRound(pad_h + 0.1);
    int left = cvRound(pad_w - 0.1);
    int right = cvRound(pad_w + 0.1);
    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return img;
}

at::Tensor preprocess_image(cv::Mat img, int imgsz=640, int stride=32) {
    img = scale_and_pad_image(img, imgsz, stride);

    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);

    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor / 255.0;

    img_tensor = img_tensor.to(at::kCUDA);

    return img_tensor;
}

at::Tensor process_detections(at::Tensor pred, float conf_thres=0.25, int num_masks=0) {
    // compute the conf score for each class - conf = obj_conf * cls_conf
    // in python this would be done with           output[:, 5:] *= output[:, 4:5]
    // but in C++ we need to do it like this:
    int num_classes = pred.size(1) - 5 - num_masks;

    auto obj_conf = pred.slice(1, 4, 5);
    auto cls_conf = pred.slice(1, 5, 5 + num_classes);
    cls_conf = cls_conf * obj_conf.expand_as(cls_conf);

    // get the maximum class score for each box
    tuple<at::Tensor, at::Tensor> tuple = torch::max(cls_conf, 1);
    at::Tensor max_conf = get<0>(tuple);
    at::Tensor max_conf_idx = get<1>(tuple);

    at::Tensor detections = torch::zeros({pred.size(0), 6 + num_masks}).to(torch::kCUDA);

    detections.slice(1, 0, 4) = pred.slice(1, 0, 4);
    detections.slice(1, 4, 5) = max_conf.unsqueeze(1);
    detections.slice(1, 5, 6) = max_conf_idx.unsqueeze(1);

    if (num_masks > 0) {
        detections.slice(1, 6, 6 + num_masks) = pred.slice(1, 5 + num_classes, 5 + num_classes + num_masks);
    }

    // filter out the boxes with low conf scores
    auto keep = (max_conf > conf_thres).nonzero().squeeze();
    detections = detections.index_select(0, keep);

    // convert the boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2]
    auto x1 = detections.slice(1, 0, 1) - detections.slice(1, 2, 3) / 2;
    auto y1 = detections.slice(1, 1, 2) - detections.slice(1, 3, 4) / 2;
    auto x2 = detections.slice(1, 0, 1) + detections.slice(1, 2, 3) / 2;
    auto y2 = detections.slice(1, 1, 2) + detections.slice(1, 3, 4) / 2;

    // add the boxes to the output tensor
    detections = torch::cat({x1, y1, x2, y2, detections.slice(1, 4, detections.size(1))}, 1);
    return detections;
}

std::vector<int> non_max_suppression(at::Tensor detections, float conf_thres=0.25, float iou_thres=0.45) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < detections.size(0); i++) {
        auto x1 = detections[i][0].item<float>();
        auto y1 = detections[i][1].item<float>();
        auto x2 = detections[i][2].item<float>();
        auto y2 = detections[i][3].item<float>();
        boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(detections[i][4].item<float>());
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres, iou_thres, indices);

    return indices;
}

void clip_boxes(at::Tensor& detections, int img_w, int img_h) {
    // clip boxes to image bounds
    detections.index({at::indexing::Slice(), 0}).clamp_(0, img_w);
    detections.index({at::indexing::Slice(), 1}).clamp_(0, img_h);
    detections.index({at::indexing::Slice(), 2}).clamp_(0, img_w);
    detections.index({at::indexing::Slice(), 3}).clamp_(0, img_h);
}

void scale_boxes(at::Tensor& detections, double im1_w, double im1_h, double im0_w, double im0_h) {
    // transform box coordinates back to the coordinate space of the original image
    double upscale = std::max(im0_w / im1_w, im0_h / im1_h);

    double pad_w = (im1_w - im0_w / upscale) / 2;
    double pad_h = (im1_h - im0_h / upscale) / 2;

    detections.index({at::indexing::Slice(), 0}) = (detections.index({at::indexing::Slice(), 0}) - pad_w) * upscale;
    detections.index({at::indexing::Slice(), 1}) = (detections.index({at::indexing::Slice(), 1}) - pad_h) * upscale;
    detections.index({at::indexing::Slice(), 2}) = (detections.index({at::indexing::Slice(), 2}) - pad_w) * upscale;
    detections.index({at::indexing::Slice(), 3}) = (detections.index({at::indexing::Slice(), 3}) - pad_h) * upscale;

    clip_boxes(detections, im0_w, im0_h);
}

void print_mask(at::Tensor mask, int step = 3) {
    for (int i = 0; i < mask.size(0); i += step) {
        for (int j = 0; j < mask.size(1); j += step) {
            if (mask[i][j].item<float>() > 0.5) {
                cout << "1 ";
            } else {
                cout << "0 ";
            }
        }
        cout << endl;
    }
}

at::Tensor scale_masks(at::Tensor& masks, int im1_w, int im1_h, int im0_w, int im0_h) {
    double upscale = std::max((double)im0_w / im1_w, (double)im0_h / im1_h);

    int pad_w = (im1_w - im0_w / upscale) / 2;
    int pad_h = (im1_h - im0_h / upscale) / 2;

    // scale the masks to the size of the original image
    int new_w = im1_w * upscale;
    int new_h = im1_h * upscale;

    at::Tensor upscaled_masks = F::interpolate(masks.unsqueeze(1), F::InterpolateFuncOptions().size(std::vector<int64_t>({new_h, new_w})).mode(torch::kBilinear).align_corners(false));
    upscaled_masks = upscaled_masks.squeeze(1);

    for (int i = 0; i < upscaled_masks.size(0); i++) {
        print_mask(upscaled_masks[i], 20);
    }

    // crop the masks to the original image
    upscaled_masks = upscaled_masks.index({at::indexing::Slice(), at::indexing::Slice(pad_h, pad_h + im0_h), at::indexing::Slice(pad_w, pad_w + im0_w)});

    return upscaled_masks;

}

void scale_segments(vector<vector<cv::Point>>& segments, double im1_w, double im1_h, double im0_w, double im0_h) {
    double upscale = std::max(im0_w / im1_w, im0_h / im1_h);

    double pad_w = (im1_w - im0_w / upscale) / 2;
    double pad_h = (im1_h - im0_h / upscale) / 2;

    for (int i = 0; i < segments.size(); i++) {
        for (int j = 0; j < segments[i].size(); j++) {
            segments[i][j].x = (segments[i][j].x - pad_w) * upscale;
            segments[i][j].y = (segments[i][j].y - pad_h) * upscale;
        }
    }
}


at::Tensor crop_masks(at::Tensor masks, at::Tensor detections) {
    at::Tensor cropped_masks = torch::zeros({masks.size(0), masks.size(1), masks.size(2)});

    for (int i = 0; i < detections.size(0); i++) {
        int x1 = detections[i][0].item<float>();
        int y1 = detections[i][1].item<float>();
        int x2 = detections[i][2].item<float>();
        int y2 = detections[i][3].item<float>();

        cropped_masks[i].index({at::indexing::Slice(y1, y2), at::indexing::Slice(x1, x2)}) = masks[i].index({at::indexing::Slice(y1, y2), at::indexing::Slice(x1, x2)});
    }

    return cropped_masks;
}

at::Tensor process_masks(at::Tensor protos, at::Tensor detections, int im1_w, int im1_h) {
    int mh = protos.size(1);
    int mw = protos.size(2);

    int mask_index = detections.size(1) - 32;
    auto mask_emb = detections.slice(1, mask_index, mask_index + 32);

    // for each pixel compute the dot-product between the mask embedding and the mask prototype
    // masks.shape = (num_detections, mh, mw)
    auto masks = mask_emb.mm(protos.view({protos.size(0), -1})).sigmoid().view({-1, mh, mw});

    // upsample the masks to the size of the original image
    torch::Tensor upscaled_masks = F::interpolate(masks.unsqueeze(0), F::InterpolateFuncOptions().size(std::vector<int64_t>({im1_h, im1_w})).mode(torch::kBilinear).align_corners(false)).squeeze(0);

    upscaled_masks = crop_masks(upscaled_masks, detections);

    return upscaled_masks;
}


vector<vector<cv::Point>> masks_to_segments(at::Tensor masks) {
     vector<vector<cv::Point>> mask_segments = vector<vector<cv::Point>>();

     masks = (masks > 0.5) * 255;

    for (int i = 0; i < masks.size(0); i++) {
        at::Tensor mask = masks[i].to(torch::kU8).to(torch::kCPU);
        cv::Mat cv_mask = cv::Mat(mask.size(0), mask.size(1), CV_8UC1);
        // Copy data from tensor to cv::Mat
        memcpy(cv_mask.data, mask.data_ptr(), sizeof(torch::kU8) * mask.numel());
        vector<vector<cv::Point>> contours = vector<vector<cv::Point>>();
        cv::findContours(cv_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Select the largest contour
        int max_contour_index = 0;
        for (int j = 0; j < contours.size(); j++) {
            if (contours[j].size() > contours[max_contour_index].size()) {
                max_contour_index = j;
            }
        }

        if (contours.size() == 0) {
            mask_segments.push_back(vector<cv::Point>());
        } else {
            mask_segments.push_back(contours[max_contour_index]);
        }
        // Add the largest contour to the mask segments
    }

    return mask_segments;
}




torch::jit::script::Module load_model(const std::string& model_path) {
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.to(at::kCUDA);
    model.eval();
    return model;
}


cv::Scalar colorPallete(int i) {
    std::vector<cv::Scalar> colors = {cv::Scalar(255, 56, 56), cv::Scalar(255, 157, 151), cv::Scalar(255, 112, 31), cv::Scalar(255, 178, 29), cv::Scalar(207, 210, 49), cv::Scalar(72, 249, 10), cv::Scalar(146, 204, 23), cv::Scalar(61, 219, 134), cv::Scalar(26, 147, 52), cv::Scalar(0, 212, 187), cv::Scalar(44, 153, 168), cv::Scalar(0, 194, 255), cv::Scalar(52, 69, 147), cv::Scalar(100, 115, 255), cv::Scalar(0, 24, 236), cv::Scalar(132, 56, 255), cv::Scalar(82, 0, 133), cv::Scalar(203, 56, 255), cv::Scalar(255, 149, 200), cv::Scalar(255, 55, 199)};

    return colors[i % colors.size()];
}

void draw_bounding_boxes(at::Tensor& detections, cv::Mat& img) {
    for (int i = 0; i < detections.size(0); i++) {
        auto x1 = detections[i][0].item<float>();
        auto y1 = detections[i][1].item<float>();
        auto x2 = detections[i][2].item<float>();
        auto y2 = detections[i][3].item<float>();
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);

        // Draw bounding box
        int class_id = detections[i][5].item<int>();
        cv::rectangle(img, box, colorPallete(class_id), 2);


        // Draw class label and confidence
        // First draw a rectangle to put the label
        cv::rectangle(img, cv::Point(box.x, box.y - 30), cv::Point(box.x + 100, box.y), colorPallete(class_id), -1);


        std::string label = std::to_string(class_id) + ": " + std::to_string(detections[i][4].item<float>());
        cv::putText(img, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }
}


void print_cv_mask(cv::Mat mask, int step = 10) {
    // mask type: CV_8UC1

    for (int i = 0; i < mask.rows; i += step) {
        for (int j = 0; j < mask.cols; j += step) {
            if (mask.at<uchar>(i, j) > 127) {
                cout << "1 ";
            } else {
                cout << "0 ";
            }
        }
        cout << endl;
    }
}


void print_sum_mask(at::Tensor sum_mask, int step=20) {
    for (int i = 0; i < sum_mask.size(0); i += step) {
        for (int j = 0; j < sum_mask.size(1); j += step) {
            int sum_color = 0;

            for (int k = 0; k < 3; k++) {
                sum_color += sum_mask[i][j][k].item<float>();
            }

            if (sum_color > 0) {
                cout << "1 ";
            } else {
                cout << "0 ";
            }

        }
        cout << endl;
    }

}


void draw_masks(at::Tensor& detections, at::Tensor& masks, cv::Mat& img, bool soft_mask = true) {
    at::Tensor masks_with_color;

    if (!soft_mask) {
        masks = masks.gt_(0.5);
    }

    masks_with_color = masks.unsqueeze(1).repeat({1, 3, 1, 1}).to(at::kCPU);

    for (int i = 0; i < masks.size(0); i++) {
        cv::Scalar color = colorPallete(detections[i][5].item<int>());

        // convert color to tensor
        at::Tensor color_tensor = torch::zeros({3});
        color_tensor[0] = color[0];
        color_tensor[1] = color[1];
        color_tensor[2] = color[2];

        // multiply mask with color
        // masks_with_color[i].shape = (3, mh, mw)
        // color_tensor.shape = (3)

        masks_with_color[i] = masks_with_color[i] * color_tensor.view({3, 1, 1});
    }

    at::Tensor sum_mask = masks_with_color.sum(0).clamp_(0, 255).to(torch::kU8).permute({1, 2, 0}).contiguous();

    // Draw mask
    cv::Mat mask_mat = cv::Mat(sum_mask.size(0), sum_mask.size(1), CV_8UC3);
    // Copy data from tensor to cv::Mat
    memcpy((void *)mask_mat.data, sum_mask.data_ptr(), sizeof(torch::kU8) * sum_mask.numel());

    cv::addWeighted(img, 1, mask_mat, 1, 0.0, img);
}


struct cmd_args {
    string model_path;
    float conf_thresh;
    float iou_thresh;
    int img_size;
};

void parse_args(int argc, char *argv[], cmd_args &args)
{
    // default values
    args.model_path = "model.pt";
    args.conf_thresh = 0.5;
    args.iou_thresh = 0.5;
    args.img_size = 640;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--model-path") == 0)
        {
            args.model_path = argv[i+1];
        }
        else if (strcmp(argv[i], "--conf-thresh") == 0)
        {
            args.conf_thresh = atof(argv[i+1]);
        }
        else if (strcmp(argv[i], "--iou-thresh") == 0)
        {
            args.iou_thresh = atof(argv[i+1]);
        }
        else if (strcmp(argv[i], "--imgsz") == 0)
        {
            args.img_size = atoi(argv[i+1]);

            // check that the image size is divisible by 32 (stride)
            if (args.img_size % 32 != 0)
            {
                cout << "Image size must be divisible by 32" << endl;
                exit(1);
            }
        }
    }
}