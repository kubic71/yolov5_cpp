# simple python script to draw segments

import argparse
from PIL import Image, ImageDraw

class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)

    args = parser.parse_args()

    with open(args.txt, "r") as f:
        pred = f.read().strip().split("\n")

    

    img = Image.open(args.img)

    for line in pred:
        line = line.strip().split(" ")
        cls_idx, conf, x1, y1, x2, y2 = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])

        segments = list(map(float, line[6:]))
        segments.append(segments[0])
        segments.append(segments[1])

        # draw bounding box
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=colors(cls_idx), width=3)
        label = f"{cls_idx} {conf:.2f}"
        # draw label on top of a small rectangle as a background
        draw.rectangle([x1, y1, x1 + 100, y1 + 20], fill=colors(cls_idx))
        draw.text((x1+5, y1+5), label, fill="white")

        # draw segments
        draw.line(segments, fill=colors(cls_idx), width=3)

    # show image
    img.show()


