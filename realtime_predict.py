from ultralytics import YOLO

model = YOLO('gta.pt')# I got this file when i trained my model using train.py  file from 
                      # runsFolder/detect/train/weight/ named as best.pt file which i renamed with gta.pt

results = model.predict(source="video.mp4", save=True)
