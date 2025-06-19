import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load('potato_leaf_disease_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image) -> str:
    img = transform(image).unsqueeze(0)
    model = load_model()
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

def main():
    st.set_page_config(page_title='Potato Leaf Disease Detector', layout='wide')
    st.title('Potato Leaf Disease Detector')
    st.markdown('Upload a picture of a potato leaf and discover possible diseases.')

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=False, width=400)

        if st.button('Predict'):  # interactive button
            label = predict(image)
            st.success(f'Detected Disease: **{label}**')

if __name__ == '__main__':
    main()
