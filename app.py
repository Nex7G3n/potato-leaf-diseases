import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import json

CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus']

@st.cache_resource
def load_models():
    models_dict = {}

    # Load ResNet18
    resnet18_model = models.resnet18(pretrained=False)
    resnet18_model.fc = torch.nn.Linear(resnet18_model.fc.in_features, len(CLASS_NAMES))
    resnet18_state_dict = torch.load('models/potato_leaf_disease_model_resnet18.pth', map_location='cpu')
    resnet18_model.load_state_dict(resnet18_state_dict)
    resnet18_model.eval()
    models_dict['ResNet18'] = resnet18_model

    # Load ResNet50
    resnet50_model = models.resnet50(pretrained=False)
    resnet50_model.fc = torch.nn.Linear(resnet50_model.fc.in_features, len(CLASS_NAMES))
    resnet50_state_dict = torch.load('models/potato_leaf_disease_model_resnet50.pth', map_location='cpu')
    resnet50_model.load_state_dict(resnet50_state_dict)
    resnet50_model.eval()
    models_dict['ResNet50'] = resnet50_model

    # Load DenseNet121
    densenet121_model = models.densenet121(pretrained=False)
    densenet121_model.classifier = torch.nn.Linear(densenet121_model.classifier.in_features, len(CLASS_NAMES))
    densenet121_state_dict = torch.load('models/potato_leaf_disease_model_densenet121.pth', map_location='cpu')
    densenet121_model.load_state_dict(densenet121_state_dict)
    densenet121_model.eval()
    models_dict['DenseNet121'] = densenet121_model

    return models_dict

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model: torch.nn.Module):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item() * 100
    return predicted_class_name, confidence, probabilities.tolist()

def main():
    st.set_page_config(page_title='Detector de Enfermedades de la Hoja de Patata', layout='wide')
    st.title('Detector de Enfermedades de la Hoja de Patata')

    st.sidebar.title('Navegación')
    page = st.sidebar.radio(
        'Ir a',
        ['Inicio y Modelos', 'Resultados de Evaluación', 'Predicción de Imagen']
    )

    models_dict = load_models()

    if page == 'Inicio y Modelos':
        st.header('Bienvenido 👋 y Información General 📚')
        st.markdown("""
            Esta aplicación utiliza modelos de aprendizaje profundo para detectar enfermedades en las hojas de patata.
            Simplemente sube una imagen de una hoja de patata y nuestros modelos te ayudarán a identificar posibles enfermedades.
            Nuestro objetivo es proporcionar una herramienta útil para agricultores y entusiastas de la agricultura para
            identificar rápidamente problemas en los cultivos y tomar medidas oportunas.
        """)
        st.info("Explora las secciones a continuación para conocer más sobre el dataset y los modelos.")

        st.markdown("---") # Separador

        st.subheader('Información del Dataset')
        st.markdown("""
            Los modelos fueron entrenados con el **"Potato Leaf Disease Dataset in Uncontrolled Environment"**.
            Este dataset contiene imágenes de hojas de patata clasificadas en diferentes categorías de enfermedades y hojas sanas.
            A continuación, se muestra la distribución de las clases en el dataset:
        """)
        dataset_info = {
            "Bacteria": 569,
            "Fungi": 748,
            "Healthy": 201,
            "Nematode": 68,
            "Pest": 611,
            "Phytophthora": 347,
            "Virus": 532
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribución de Clases:")
            st.json(dataset_info)
        with col2:
            labels = list(dataset_info.keys())
            sizes = list(dataset_info.values())
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax1.axis('equal')
            st.pyplot(fig1)
            plt.close(fig1)

        st.markdown("---") # Separador

        st.subheader('Modelos de Red Neuronal Utilizados')
        st.markdown("""
            Hemos empleado tres arquitecturas de redes neuronales convolucionales (CNN) de última generación
            para la clasificación de enfermedades en hojas de patata:
        """)

        model_names = ['ResNet18', 'ResNet50', 'DenseNet121']
        model_descriptions = {
            'ResNet18': """
                **ResNet18** es una versión más ligera de la familia ResNet (Residual Networks).
                Estas redes introducen conexiones de salto (skip connections) que permiten que el gradiente
                fluya directamente a través de múltiples capas, lo que ayuda a entrenar redes muy profundas
                sin problemas de desvanecimiento del gradiente. ResNet18 es conocida por su eficiencia y buen rendimiento.
            """,
            'ResNet50': """
                **ResNet50** es una versión más profunda de ResNet que ResNet18. Utiliza más capas y bloques residuales,
                lo que le permite aprender características más complejas y, a menudo, lograr una mayor precisión
                en tareas de clasificación de imágenes, aunque con un mayor costo computacional.
            """,
            'DenseNet121': """
                **DenseNet121** (Densely Connected Convolutional Networks) es una arquitectura que conecta
                cada capa con todas las capas posteriores en una moda de "alimentación hacia adelante".
                Esto significa que la entrada de cada capa consiste en la salida de todas las capas anteriores,
                lo que fomenta la reutilización de características y reduce el número de parámetros,
                mejorando la propagación de la información y el gradiente.
            """
        }

        for model_name in model_names:
            with st.expander(f"Detalles de {model_name}"):
                st.markdown(model_descriptions[model_name])

    elif page == 'Resultados de Evaluación':
        st.header('Resultados de Evaluación de los Modelos 📊')
        st.markdown('Aquí puedes ver las métricas y gráficos de rendimiento de los modelos entrenados.')

        results_dir = 'results'
        
        # Mostrar la matriz de confusión y el informe de clasificación para cada modelo
        model_results_info = {
            'ResNet18': {
                'confusion_matrix': 'confusion_matrix_resnet18.png',
                'classification_report': 'classification_report_resnet18.txt'
            },
            'ResNet50': {
                'confusion_matrix': 'confusion_matrix_resnet50.png',
                'classification_report': 'classification_report_resnet50.txt'
            },
            'DenseNet121': {
                'confusion_matrix': 'confusion_matrix_densenet121.png',
                'classification_report': 'classification_report_densenet121.txt'
            }
        }

        for model_name, files in model_results_info.items():
            with st.expander(f"Resultados Detallados para {model_name}"):
                st.subheader(f"Matriz de Confusión para {model_name}:")
                cm_path = os.path.join(results_dir, files['confusion_matrix'])
                if os.path.exists(cm_path):
                    st.image(cm_path, caption=f'Matriz de Confusión {model_name}')
                else:
                    st.info(f"No se encontró la matriz de confusión para {model_name}.")

                st.subheader(f"Informe de Clasificación para {model_name}:")
                cr_path = os.path.join(results_dir, files['classification_report'])
                if os.path.exists(cr_path):
                    with open(cr_path, 'r') as f:
                        st.text(f.read())
                else:
                    st.info(f"No se encontró el informe de clasificación para {model_name}.")
                
                # También mostrar el JSON de evaluación si existe

        st.markdown("---") # Separador

        st.subheader('Otros Gráficos de Rendimiento:')
        # Filtrar los archivos de imagen que no son matrices de confusión específicas de modelo
        all_image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        specific_cm_files = [files['confusion_matrix'] for files in model_results_info.values()]
        other_image_files = [f for f in all_image_files if f not in specific_cm_files]

        for img_file in sorted(other_image_files):
            st.image(os.path.join(results_dir, img_file), caption=img_file.replace('_', ' ').replace('.png', ''))


    elif page == 'Predicción de Imagen':
        st.header('Predicción de Enfermedades de la Hoja de Patata 🔍')
        st.markdown('Sube una imagen de una hoja de patata para que los modelos realicen una predicción.')

        uploaded_file = st.file_uploader('Elige una imagen', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Imagen Cargada', use_column_width=False, width=400)

            if st.button('Realizar Predicción'):
                st.subheader('Resultados de la Predicción:')
                
                cols = st.columns(len(models_dict)) # Crear columnas dinámicamente

                for i, (model_name, model_obj) in enumerate(models_dict.items()):
                    with cols[i]:
                        st.markdown(f"### {model_name}")
                        predicted_class, confidence, probabilities = predict(image, model_obj)
                        
                        st.markdown(f"**Enfermedad Detectada:** **{predicted_class}**")
                        st.markdown(f"<p style='color:green; font-size:20px;'>**Confianza: {confidence:.2f}%**</p>", unsafe_allow_html=True)

                        # Gráfico de barras para las probabilidades
                        fig, ax = plt.subplots(figsize=(6, 4)) # Ajustar tamaño para columnas
                        ax.bar(CLASS_NAMES, probabilities, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'brown', 'pink'])
                        ax.set_ylabel('Probabilidad')
                        ax.set_title(f'Probabilidades de Clase')
                        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8) # Ajustar tamaño de fuente
                        ax.tick_params(axis='y', labelsize=8)
                        st.pyplot(fig)
                        plt.close(fig) # Cierra la figura para evitar que se muestre dos veces

if __name__ == '__main__':
    main()
