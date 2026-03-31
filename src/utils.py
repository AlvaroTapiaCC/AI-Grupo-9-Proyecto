import cv2

def load_image(path):
    return cv2.imread(path)

def load_labels(label_path):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            boxes.append((xc, yc, bw, bh))
    return boxes

def yolo_to_bbox(box, img_w, img_h):
    xc, yc, bw, bh = box

    x_center = xc * img_w
    y_center = yc * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    return x1, y1, x2, y2
