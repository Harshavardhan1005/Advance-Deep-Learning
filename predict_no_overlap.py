import cv2
import os
import math
import darknet
import argparse
import itertools

BEE_COLOUR = (255, 177, 1) 

def abs_bb(coords, top_left, bot_right):
    return ((coords[0][0]+top_left[0],
            coords[0][1]+top_left[1]),
            (coords[1][0]+top_left[0],
            coords[1][1]+top_left[1]))

def bbox2points(bbox):
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return ((xmin, ymin), (xmax, ymax))

def detect_frame(frame, network, class_names, thresh):
    detects = []
    darknet.copy_image_from_bytes(darknet_image, frame.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    for x in detections:
        tl, br = bbox2points(x[2])
        detects.append((float(x[1]), tl, br))
    return detects

def is_inside(bb1, bb2):
    if (bb2[1][0] <= bb1[1][0]) and (bb1[2][0] <= bb2[2][0]):
        if (bb2[1][1] <= bb1[1][1]) and (bb1[2][1] <= bb2[2][1]):
            return bb1
    elif (bb1[1][0] <= bb2[1][0]) and (bb2[2][0] <= bb1[2][0]):
        if (bb1[1][1] <= bb2[1][1]) and (bb2[2][1] <= bb1[2][1]):
            return bb2
    return 0

def stats(truth_boxes, pred_boxes, thresh=0.5):
    tp = fn = fp = 0
    if not pred_boxes and truth_boxes:
        return 0, 0
    if not truth_boxes and pred_boxes:
        return 0, 0
    for pred in pred_boxes:
        if any(iou(pred, truth) > thresh for truth in truth_boxes):
            tp += 1
        else:
            fp += 1
    for truth in truth_boxes:
        if all(iou(pred, truth) < thresh for pred in pred_boxes):
            fn += 1
    return tp/(tp+fp), tp/(tp+fn)


def iou(bb1, bb2):
    x_left = max(bb1[1][0], bb2[1][0])
    y_top = max(bb1[1][1], bb2[1][1])
    x_right = min(bb1[2][0], bb2[2][0])
    y_bottom = min(bb1[2][1], bb2[2][1])
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2][0] - bb1[1][0]) * (bb1[2][1] - bb1[1][1])
    bb2_area = (bb2[2][0] - bb2[1][0]) * (bb2[2][1] - bb2[1][1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    return iou

def remove_overlapped(detects, drop_threshold=0.3):
    combos = itertools.combinations(detects, 2)
    result = detects
    for bb1, bb2 in combos:
        if bb1 in result and bb2 in result:
            inside = is_inside(bb1, bb2)
            if inside:
                result.remove(inside)
            elif iou(bb1, bb2) > drop_threshold:
                result.remove(bb1 if bb1[0] < bb2[0] else bb2)
    return result

def keep_continuous(latest_detects, thresh):
    kept = []
    for bb in latest_detects[-1]:
        if all(any(iou(i, bb) > thresh for i in x) for x in latest_detects[:-1]):
            kept.append(bb)
    return kept

def make_frame(frame, detects):
    for confidence, tl, br in detects:
        cv2.rectangle(frame, tl, br, (255, 177, 1), 3)
        cv2.putText(frame, "{} [{:.2f}]".format("Bee", confidence),
                (tl[0], tl[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                BEE_COLOUR, 2)
    return frame
    #cv2.waitKey(0)
def parser():
    parser = argparse.ArgumentParser(description="Bee detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source.")
    parser.add_argument("--out_filename", type=str, default="output.mp4",
                        help="inference video name.")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.5,
                        help="remove detections with confidence below this value")
    parser.add_argument("--show", type=bool, default=False,
                        help="show the training process in a window")
    parser.add_argument("--truth", type=bool, default=False,
                        help="show the truth boxes")
    parser.add_argument("--filter", type=bool, default=False,
                        help="filter the bbs")
    return parser.parse_args()

def mean(l):
    return sum(l)/len(l)

if __name__ == '__main__':
    args = parser()
    if not args.out_filename:
        args.out_filename = "predicted.mp4"

    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
        )

    print(args.input)
    cap = cv2.VideoCapture(args.input)
    res, frame = cap.read()
    
    h, w = frame.shape[:2]
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(w, h, 3)
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_filename, fourcc, fps, (w,  h))
    if args.show:
        cv2.namedWindow('bees',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('bees', round(w/2), round(h/2))

    # keep a running tally of our latest 3 detections
    latest_detects = []
    detect_count = 0
    yseries = []
    frame_no = 0
    mAPs, recalls = [], []
    seen = False
    while res:
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        f_name = f"/content/gdrive/My Drive/labels/img2_{frame_no:03}.txt"
        truth_boxes = []
        if (os.path.isfile(f_name)):
            with open(f_name) as f:
                lines = f.readlines()
                for line in lines:
                    splt = [float(x) for x in line.strip().split()]
                    truth_boxes.append(bbox2points((splt[1]*w, splt[2]*h, splt[3]*w, splt[4]*h)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detects = detect_frame(frame_rgb, network, class_names, args.thresh)
        removed = remove_overlapped(detects)
        if args.filter:
            latest_detects.append(removed)
            if len(latest_detects) > 2:
                removed = keep_continuous(latest_detects, 0.1)
                latest_detects.pop(0)
        if len(removed) > 0:
            seen = True
        if frame_no % fps == 0:
            if seen:
                yseries.append(1)
            else:
                yseries.append(0)
            seen = False

        annotated = make_frame(frame_rgb, removed)
        if args.truth:
            for p in truth_boxes:
                cv2.rectangle(annotated, p[0], p[1], (255, 0, 0), 3)
                precision, recall = stats([(0, x[0], x[1]) for x in truth_boxes], removed)
                mAPs.append(precision)
                recalls.append(recall)
                cv2.putText(annotated, f"mAP: {mean(mAPs)*100:.2f}% mRecall: {mean(recalls)*100:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        if args.show:
            cv2.imshow('bees', cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        res, frame = cap.read()
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    yseries.append(detect_count/fps)
    if args.truth:
        print(mAPs, recalls)
        print(f"mAP: {sum(mAPs)/len(mAPs)} mRecall: {sum(recalls)/len(recalls)}")
    with open("detections.txt", 'w') as f:
        f.write('\n'.join(map(str, yseries)))
