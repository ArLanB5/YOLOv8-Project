import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live object detection")
    parser.add_argument(
        "--web-resolution", 
        type=int, 
        default=[1280, 720],
        nargs=2,
        help="Resolution of the webcam feed"
    )
    args = parser.parse_args()
    return args

def draw_rounded_rectangle(image, top_left, bottom_right, color, radius, thickness):
    if image is None:
        raise ValueError("The input image is None. Please provide a valid image.")
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1
    if radius > min(w, h) // 2:
        radius = min(w, h) // 2

    overlay = image.copy()

    # Draw four corners
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)

    # Draw four edges
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y1 + radius), color, -1)
    cv2.rectangle(overlay, (x1 + radius, y2 - radius), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
    cv2.rectangle(overlay, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)

    # Draw center rectangle
    cv2.rectangle(overlay, (x1 + radius, y1 + radius), (x2 - radius, y2 - radius), color, -1)

    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

def get_text_size(text, font, font_scale, thickness):
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    return size

def main():
    args = parse_arguments()
    frame_width, frame_height = args.web_resolution

    ZONE_POLYGON = np.array([
        [0, 0],
        [frame_width, 0],
        [frame_width, frame_height],
        [0, frame_height]
    ])
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1.0
    )

    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.web_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    def get_price(item):
        prices = {
            "person": 0,
            "bicycle": 5000,
            "car": 600000,
            "motorcycle": 120000,
            "bus": 1000000,
            "truck": 1200000,
            "backpack": 4000,
            "umbrella": 500,
            "handbag": 2000,
            "tie": 1500,
            "suitcase": 5000,
            "baseball bat": 2000,
            "baseball glove": 800,
            "skateboard": 1800,
            "surfboard": 2000,
            "tennis racket": 2500,
            "bottle": 30,
            "cup": 500,
            "fork": 100,
            "knife": 200,
            "spoon": 250,
            "bowl": 1000,
            "banana": 80,
            "apple": 160,
            "sandwich": 50,
            "orange": 150,
            "broccoli": 80,
            "carrot": 40,
            "hot dog": 80,
            "pizza": 200,
            "donut": 160,
            "cake": 7000,
            "chair": 1500,
            "couch": 7000,
            "potted plant": 400,
            "tv": 30000,
            "laptop": 80000,
            "mouse": 4000,
            "remote":300,
            "keyboard": 5000,
            "cell phone": 25000,
            "microwave": 35000,
            "oven": 50000,
            "toaster": 2500,
            "refrigerator": 16000,
            "book": 1200,
            "clock": 600,
            "vase": 250,
            "scissors": 40,
            "teddy bear": 1800,
            "hair drier": 1500,
            "toothbrush": 30
        }
        return prices.get(item, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
       
        labels = [
            f"{model.model.names[class_id]}"
            for _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        total_objects = len(detections)
        total_amount = sum(get_price(model.model.names[class_id]) for _, _, class_id, _ in detections)

        # Draw rounded rectangle and text for total items detected
        items_text = f"TOTAL ITEMS DETECTED: {total_objects}"
        items_text_size, _ = cv2.getTextSize(items_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        items_box_width = items_text_size[0] + 20
        items_box_height = items_text_size[1] + 20
        items_box_top_left = (20, 20)
        items_box_bottom_right = (20 + items_box_width, 20 + items_box_height)
        draw_rounded_rectangle(frame, items_box_top_left, items_box_bottom_right, (139, 0, 0), 10, 2)
        cv2.putText(
            frame,
            items_text,
            (items_box_top_left[0] + 10, items_box_top_left[1] + items_text_size[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw rounded rectangle and text for total amount
        amount_text = f"TOTAL AMOUNT: Rs {total_amount}"
        amount_text_size, _ = cv2.getTextSize(amount_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        amount_box_width = amount_text_size[0] + 20
        amount_box_height = amount_text_size[1] + 20
        amount_box_top_left = (20, items_box_bottom_right[1] + 10)
        amount_box_bottom_right = (20 + amount_box_width, items_box_bottom_right[1] + 10 + amount_box_height)
        draw_rounded_rectangle(frame, amount_box_top_left, amount_box_bottom_right, (139, 0, 0), 10, 2)
        cv2.putText(
            frame,
            amount_text,
            (amount_box_top_left[0] + 10, amount_box_top_left[1] + amount_text_size[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        y_offset = 30
        item_counts = {}
        for label in labels:
            if label in item_counts:
                item_counts[label] += 1
            else:
                item_counts[label] = 1

        unique_labels = set()
        text = "DETECTED ITEMS: "
        size3 = get_text_size(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        detected_items_height = 50 + len(item_counts) * 35
        draw_rounded_rectangle(frame, (frame_width - 310, 10), (frame_width - 10, 10 + detected_items_height), (139, 0, 0), 10, 2)

        cv2.putText(
            frame, 
            text, 
            (frame_width - 300, y_offset + size3[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        y_offset += 40

        for i, (label, count) in enumerate(item_counts.items(), start=1):
            if label not in unique_labels:
                unique_labels.add(label)
                price = get_price(label)
                item_text = f"{i}. {label} x{count}: Rs {price * count}"
                size_item = get_text_size(item_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(
                    frame, 
                    item_text, 
                    (frame_width - 300, y_offset + size_item[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2, 
                    cv2.LINE_AA
                )
                y_offset += 30

        cv2.imshow("yolov8", frame)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
