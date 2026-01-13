import torch
from torchvision import transforms
import cv2
from PIL import Image

from ml.model import GestureCNN
import ml.config as config

class GesturePredictor():

    def __init__(self, model, path_to_trained_model, device):

        self.model = model
        self.device = device
        self.model.load_state_dict(torch.load(path_to_trained_model, map_location=self.device))
    
    def preprocess_image(self, frame, bbox, mean_norm, std_norm, img_size):
        """
        Get an OpenCV numpy frame, cropout out region of the hand, resize it and return as a Normalized Tensor
        """

        #Here, the input parameter frame will be an OpenCV (i.e. BGR) Numpy array of pixels. 
        #This needs to be processed and returned in the tensor format the model expects.

        x_min, y_min, x_max, y_max = bbox
        frame_cropped = frame[x_min:x_max, y_min:y_max]
        frame_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_cropped, (img_size, img_size))

        pil_image = Image.fromarray(frame_resized)

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_norm, std=std_norm)
        ])

        tensor = transformation(pil_image)
        tensor = tensor.unsqueeze(0) #As the model expects the batch dimension too
        tensor = tensor.to(self.device)
        return tensor
