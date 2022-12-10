// Copyright (c) 2021 Cesanta Software Limited
// All rights reserved

#include "mongoose.h"
#include <opencv2/opencv.hpp>
#include "segment.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.h"

// HTTP request handler function. It implements the following endpoints:
//   /upload - prints all submitted form elements
//   all other URI - serves web_root/ directory
//
// /////////////////           IMPORTANT        //////////////////////////
//
// Mongoose has a limit on input buffer, which also limits maximum upload size.
// It is controlled by the MG_MAX_RECV_SIZE constant, which is set by
// default to (3 * 1024 * 1024), i.e. 3 megabytes.
// Use -DMG_MAX_BUF_SIZE=NEW_LIMIT to override it.
//
// Also, consider changing -DMG_IO_SIZE=SOME_BIG_VALUE to increase IO buffer
// increment when reading data.
using namespace std;

torch::jit::script::Module model;
struct cmd_args args;

string process_img(cv::Mat cv_img)
{
    auto tmp = segm_predict(cv_img, model, args.conf_thresh, args.iou_thresh, args.img_size);
    auto detections = get<0>(tmp);
    auto segments = get<2>(tmp);

    return segm_predictions_to_str(detections, segments);
}

static void cb(struct mg_connection *c, int ev, void *ev_data, void *fn_data)
{
    if (ev == MG_EV_HTTP_MSG)
    {
        struct mg_http_message *hm = (struct mg_http_message *)ev_data;
        MG_INFO(("New request to: [%.*s], body size: %lu", (int)hm->uri.len,
                 hm->uri.ptr, (unsigned long)hm->body.len));
        if (mg_http_match_uri(hm, "/detect"))
        {
            struct mg_http_part part;
            mg_http_next_multipart(hm->body, 0, &part);
            // MG_INFO(("Chunk name: [%.*s] filename: [%.*s] length: %lu bytes", (int)part.name.len, part.name.ptr, (int)part.filename.len,
            //  part.filename.ptr, (unsigned long)part.body.len));

            // decode the image
            cv::Mat1b data(1, part.body.len, (uchar *)part.body.ptr);
            cv::Mat cv_img = cv::imdecode(data, cv::IMREAD_COLOR);

            cout << "Image size: " << cv_img.size() << endl;

            string pred = process_img(cv_img);

            mg_http_reply(c, 200, "", "%s", pred.c_str());
        }
        else
        {
            struct mg_http_serve_opts opts = {.root_dir = "."};
            mg_http_serve_dir(c, (mg_http_message *)ev_data, &opts);
        }
    }
}

int main(int argc, char *argv[])
{

    parse_args(argc, argv, args);

    try
    {
        model = load_model(args.model_path);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    struct mg_mgr mgr;

    mg_mgr_init(&mgr);
    mg_log_set(MG_LL_DEBUG); // Set log level
    mg_http_listen(&mgr, "http://localhost:8000", cb, NULL);

    for (;;)
        mg_mgr_poll(&mgr, 50);
    mg_mgr_free(&mgr);

    return 0;
}
