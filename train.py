from ultralytics import YOLO


def main():

    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        patience=8,
        device="cpu",
        workers=0,
        project="runs",
        name="helmet_detection",
        pretrained=True,
        verbose=True
    )

    print("\n===================================")
    print("Helmet Detection Training Completed!")
    print("===================================")
    print("Best model:")
    print("runs/helmet_detection/weights/best.pt")
    print("Last model:")
    print("runs/helmet_detection/weights/last.pt")


if __name__ == "__main__":
    main()