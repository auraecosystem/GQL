import argparse
import time

import cv2
import numpy as np

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.40
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

COLORS = [
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
]


def build_model(model_path, use_cuda=False):
    net = cv2.dnn.readNetFromONNX(model_path)

    if use_cuda:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("Using CUDA FP16")
        except cv2.error as exc:
            print(f"CUDA setup failed: {exc}")
            print("Falling back to CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU")

    return net


def load_classes(path):
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def detect(frame, net):
    original_height, original_width = frame.shape[:2]

    # Direct resize keeps the preprocessing and coordinate scaling consistent.
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0 / 255.0,
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False,
    )

    net.setInput(blob)
    predictions = net.forward()

    # Typical YOLOv7 ONNX output: (1, candidates, 5 + classes).
    predictions = np.asarray(predictions)

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ValueError(
            f"Unexpected YOLO output shape: {predictions.shape}"
        )

    boxes = []
    scores = []
    class_ids = []

    scale_x = original_width / INPUT_WIDTH
    scale_y = original_height / INPUT_HEIGHT

    for row in predictions:
        objectness = float(row[4])

        if objectness < CONFIDENCE_THRESHOLD:
            continue

        class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        class_score = float(class_scores[class_id])

        confidence = objectness * class_score

        if confidence < SCORE_THRESHOLD:
            continue

        center_x, center_y, width, height = map(float, row[:4])

        left = int((center_x - width / 2) * scale_x)
        top = int((center_y - height / 2) * scale_y)
        box_width = int(width * scale_x)
        box_height = int(height * scale_y)

        boxes.append([left, top, box_width, box_height])
        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
    )

    if len(indices) == 0:
        return [], [], []

    indices = np.asarray(indices).reshape(-1)

    return (
        [class_ids[i] for i in indices],
        [scores[i] for i in indices],
        [boxes[i] for i in indices],
    )


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv7 ONNX object detection"
    )
    parser.add_argument(
        "--model",
        default="config_files/yolov7.onnx",
    )
    parser.add_argument(
        "--classes",
        default="config_files/classes.txt",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video path, camera index (0), or supported stream URL",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    args = parser.parse_args()

    classes = load_classes(args.classes)
    net = build_model(args.model, args.backend == "cuda")

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    total_frames = 0
    window_frames = 0
    window_start = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("End of stream or frame read failed")
                break

            class_ids, confidences, boxes = detect(frame, net)

            for class_id, confidence, box in zip(
                class_ids, confidences, boxes
            ):
                if not 0 <= class_id < len(classes):
                    continue

                x, y, width, height = box
                color = COLORS[class_id % len(COLORS)]

                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), color, 2
                )

                label = f"{classes[class_id]}: {confidence:.2f}"
                label_y = max(y - 8, 20)

                cv2.putText(
                    frame,
                    label,
                    (max(x, 0), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            total_frames += 1
            window_frames += 1

            elapsed = time.perf_counter() - window_start

            if elapsed >= 1.0:
                fps = window_frames / elapsed
                window_frames = 0
                window_start = time.perf_counter()
            else:
                fps = window_frames / max(elapsed, 1e-9)

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow("YOLOv7 Detection", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Stopped by user")
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Total frames processed: {total_frames}")


if __name__ == "__main__":
    main()
