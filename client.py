# Simple python client that sends HTTP POST with image and receives response with predictions.

import argparse
import requests
import time
import cv2


def run(args):
    # define a video capture object
    if args.webcam:

        vid = cv2.VideoCapture(0)

        class WebcamFrame:
            frame = None

        def run_webcam(webcam_frame: WebcamFrame):
            # reads frames from a camera in a loop
            while True:
                webcam_frame.frame = vid.read()[1]

        last_frame = vid.read()[1]
        wf = WebcamFrame()
        wf.frame = last_frame

        import threading
        thread = threading.Thread(target=run_webcam, args=wf)
        thread.start()

        fps_window = 100
        response_times = [float("inf")] * fps_window

        with requests.Session() as session:
            while True:

                # TODO how to update last_frame from the thread?
                # last_frame = wf.frame
                last_frame = vid.read()[1]

                if args.compress:
                    img_bytes = cv2.imencode('.jpg', last_frame)[1].tobytes()
                else:
                    img_bytes = cv2.imencode('.bmp', last_frame)[1].tobytes()

                pred = session.post("http://" + args.host + "/" + args.mode, files={"image": img_bytes})

                print("Predictions:\n", pred.text)

                # compute average FPS
                response_times.append(time.time())
                response_times.pop(0)
                print("Average FPS:", fps_window / (response_times[-1] - response_times[0]))


    elif args.file is not None:
        with open(args.file, "rb") as f:
            img_bytes = f.read()

        pred = requests.post("http://" + args.host + "/" + args.mode, files={"image": img_bytes})
        print("Predictions:\n", pred.text)
    else:
        print("specify either --file <path> or --webcam")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost:8000")

    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of image file. Overrides --file argument.")
    parser.add_argument("--mode", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--compress", action="store_true", help="Compress image before sending. If not specified, image is sent as bmp.")


    args = parser.parse_args()

    run(args)