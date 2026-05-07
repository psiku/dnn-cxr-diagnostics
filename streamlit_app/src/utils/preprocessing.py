from torchvision import transforms
from PIL import Image
from src.constants import IMAGE_SIZE

def prepare_image_tensor(img_pil: Image.Image, channels: int = 1):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=channels),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    return transform(img_pil).unsqueeze(0)