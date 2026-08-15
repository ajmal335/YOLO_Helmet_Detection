from ultralytics import YOLO


def main():

    # Load trained helmet detection model
    model = YOLO(
        "runs/detect/runs/helmet_detection/weights/best.pt"
    )

    # Image to test
    image_path = "examples/input/helmet_6.jpeg"

    # Run prediction
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        save_txt=False,
        save_conf=False
    )

    # Print detections
    print("\n===================================")
    print("Helmet Detection Prediction")
    print("===================================")

    for result in results:

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print("No detections found.")
            continue

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = model.names[class_id]

            print(
                f"Object: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Box: "
                f"({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )

    print("\n===================================")
    print("Prediction completed!")
    print("===================================")
    print("Annotated image saved by YOLO.")


if __name__ == "__main__":
    main()