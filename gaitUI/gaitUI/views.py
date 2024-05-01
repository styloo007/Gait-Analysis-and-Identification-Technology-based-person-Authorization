from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from torchvision import transforms, models
from PIL import Image
import cv2
import torch
import os
import mediapipe as mp
from django.conf import settings
import torch.nn as nn  

# Define transforms for test image
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
num_classes = 4  # Change to 4 classes
model = models.resnet152(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Change output to 4 classes
model.load_state_dict(torch.load('Trainedmodels/best.pth', map_location=torch.device('cpu')))  # Load the model state dict
model.eval()

# Define class labels
class_labels = ['Class1', 'Class2', 'Class3', 'Class4', 'UnAuthorized']

# Define confidence threshold
confidence_threshold = 0.7  # adjust according to your needs

def index(request):
    return render(request, 'index.html')

def analyze(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image_input']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        static_image_path = os.path.join(settings.MEDIA_ROOT, filename)
        fs.save(static_image_path, uploaded_image)

        # Function to predict class for an image
        def predict_image_class(image_path):
            image = Image.open(image_path)
            # Convert image to RGB format if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Apply transformations
            image_tensor = inference_transforms(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probability, predicted = torch.max(probabilities, 1)
                max_probability = max_probability.item()
                class_index = predicted.item() if max_probability > confidence_threshold else len(class_labels) - 1  
                return class_index

        # Inference on a test image
        test_image_path = static_image_path
        predicted_class_index = predict_image_class(test_image_path)
        predicted_class = class_labels[predicted_class_index]

        # Initialize MediaPipe Pose model
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Perform pose estimation
        image = cv2.imread(test_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw skeleton on the image
        annotated_image_path = None
        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the annotated image in the 'static' folder
            static_folder = os.path.join(settings.BASE_DIR, 'static')
            annotated_image_filename = f'annotated_{uploaded_image.name}'
            annotated_image_path = os.path.join(static_folder, annotated_image_filename)
            cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Release resources
        pose.close()

        params = {'label': predicted_class, 'img_pth': uploaded_image.name, 'annotated_image_path': annotated_image_path}
        os.remove(os.path.join(settings.MEDIA_ROOT, filename))
        return render(request, 'result.html', params)

    return render(request, 'index.html')

def delete(request):
    return render(request, 'index.html')
