from ultralytics import YOLO


def main():

    # Load pretrained YOLO11 model
    model = YOLO("yolo11n.pt")

    # Validate the validation dataset
    results = model.val(
        data="data.yaml",
        split="val",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True
    )

    print("\n===================================")
    print("Dataset Check Completed!")
    print("===================================")

    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")
    print(f"Precision : {results.box.mp:.4f}")
    print(f"Recall    : {results.box.mr:.4f}")


if __name__ == "__main__":
    main()