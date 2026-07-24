
import gradio as gr
import ollama
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
model.load_state_dict(torch.load('skin_model.pth', map_location=device))
model = model.to(device)
model.eval()

classes = ["Melanoma", "Nevus", "BCC", "AK", "BKL", "DF", "VASC"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_image(image):
    if image is None:
        return "Upload an image", "No image"

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, idx = torch.max(probs, 0)

    prediction = classes[idx]
    conf_pct = confidence.item() * 100

    prompt = f"Result: {prediction} ({conf_pct:.1f}%). Give empathetic, responsible explanation and advise seeing doctor."

    try:
        resp = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': prompt}])
        explanation = resp['message']['content']
    except:
        explanation = "Gemma not responding."

    return f"**{prediction}**\nConfidence: {conf_pct:.1f}%", explanation

demo = gr.Interface(fn=analyze_image, inputs=gr.Image(type="pil"), outputs=[gr.Textbox(), gr.Textbox()],
                    title="Skin Cancer Detector with Gemma", description="Not medical advice")
demo.launch()

