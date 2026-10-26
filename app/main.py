from fastapi import FastAPI, UploadFile, File
import spacy
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
from io import BytesIO

from torchvision.utils import save_image

from app.gan_model import Generator
from app.cnn_model import SimpleCNN


app = FastAPI()


# Assignment 1: spaCy embedding model
nlp = spacy.load("en_core_web_md")


@app.get("/")
def read_root():
    return {"message": "FastAPI is working"}


@app.get("/embedding")
def get_embedding(word: str):
    doc = nlp(word)

    return {
        "word": word,
        "embedding": doc.vector.tolist()
    }


# Assignment 2: CNN classifier
model = SimpleCNN()

model.load_state_dict(
    torch.load("cnn_model.pth", map_location="cpu")
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

generator = Generator()

generator.load_state_dict(
    torch.load("generator.pth", map_location="cpu")
)

generator.eval()


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()

    image = Image.open(
        BytesIO(image_data)
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    return {
        "prediction": classes[predicted_class]
    }
@app.get("/generate-digit")
def generate_digit():
    noise = torch.randn(1, 100)

    with torch.no_grad():
        fake_image = generator(noise)

    buffer = BytesIO()

    save_image(
        fake_image,
        buffer,
        format="PNG",
        normalize=True
    )

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "image": image_base64
    }
