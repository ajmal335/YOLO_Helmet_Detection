from ultralytics import YOLO


def main():

    # Load our trained helmet detection model
    model = YOLO(
        "runs/detect/runs/helmet_detection/weights/best.pt"
    )

    # Evaluate on TEST dataset
    results = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True
    )

    print("\n===================================")
    print("Helmet Detection Test Evaluation")
    print("===================================")

    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")
    print(f"Precision : {results.box.mp:.4f}")
    print(f"Recall    : {results.box.mr:.4f}")


if __name__ == "__main__":
    main()