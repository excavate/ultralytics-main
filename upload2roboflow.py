from roboflow import Roboflow



rf = Roboflow(api_key="yvxfZDNWGHZAG6ZMKJ6z")
project = rf.workspace().project("8.7_laptop")

version = project.version(7)
version.deploy("yolov8", "../runs/detect/train52","weights/best.pt")