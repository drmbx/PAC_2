from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.eval()

out = model("man.jpg")
print(out)
