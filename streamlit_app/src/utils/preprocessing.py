from torchvision import transforms
from PIL import Image
from src.constants import IMAGE_SIZE

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def prepare_image_tensor(img_pil: Image.Image):
    return transform(img_pil).unsqueeze(0)