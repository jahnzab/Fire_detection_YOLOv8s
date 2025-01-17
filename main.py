from ultralytics import YOLO
from PIL import Image

# use directly gta.pt file or Load your best.pt File here if you trained the model by train.py file .
model = YOLO('gta.pt')

# Run the model on an image
results = model('imggfire.jpg')

# Option 1: Display the results
result_image = results[0].plot()
Image.fromarray(result_image).show()



