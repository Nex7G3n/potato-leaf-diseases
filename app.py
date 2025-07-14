import streamlit as st
import torch
from torchvision import models, transforms
torch.set_num_threads(1)
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import sys
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Preformatted, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib import colors


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
        ['Inicio y Modelos', 'Resultados de Evaluación', 'Predicción de Imagen', 'Generar Reporte PDF']
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
            Los modelos fueron entrenados con el **"Potato Leaf Disease Dataset in Uncontrolled Environment"**
            https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset .Este dataset contiene imágenes de hojas de patata clasificadas en diferentes categorías de enfermedades y hojas sanas.
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
                
                # Mostrar el Coeficiente de Correlación de Matthews (MCC)
                eval_json_path = os.path.join(results_dir, f'evaluation_results_potato_leaf_disease_model_{model_name.lower()}.json')
                if os.path.exists(eval_json_path):
                    with open(eval_json_path, 'r') as f:
                        eval_data = json.load(f)
                        if 'matthews_corrcoef' in eval_data:
                            st.subheader(f"Coeficiente de Matthews (MCC) para {model_name}:")
                            st.write(f"**MCC:** {eval_data['matthews_corrcoef']:.4f}")
                        else:
                            st.info(f"No se encontró el Coeficiente de Matthews para {model_name}.")
                else:
                    st.info(f"No se encontraron resultados de evaluación para {model_name}.")

        st.markdown("---") # Separador

        st.subheader('Resultados de la Prueba de McNemar para Comparación de Modelos:')
        mcnemar_results_path = os.path.join(results_dir, 'mcnemar_test_results.json')
        if os.path.exists(mcnemar_results_path):
            with open(mcnemar_results_path, 'r') as f:
                mcnemar_results = json.load(f)
            
            for result in mcnemar_results:
                st.markdown(f"##### Comparación: {result['model1']} vs {result['model2']}")
                if 'statistic' in result['results']:
                    st.write(f"Estadístico Chi-cuadrado: {result['results']['statistic']:.4f}")
                    st.write(f"Valor p: {result['results']['pvalue']:.4f}")
                    st.write(f"Conclusión: {result['results']['conclusion']}")
                else:
                    st.write(f"Error al realizar la prueba: {result['results']['error']}")
                st.markdown("---")
        else:
            st.info("No se encontraron resultados de la Prueba de McNemar. Ejecuta 'scripts/plot_results.py' para generarlos.")

        st.markdown("---") # Separador

        st.subheader('Otros Gráficos de Rendimiento:')

        st.subheader('Otros Gráficos de Rendimiento:')
        
        # Define image details with subtitles and descriptions
        image_details = {
            'accuracy_per_model.png': {
                'subtitle': 'Precisión por Modelo',
                'description': 'Este gráfico muestra la precisión general de cada modelo (ResNet18, ResNet50, DenseNet121) en el conjunto de datos de validación.'
            },
            'performance_comparison_accuracy.png': {
                'subtitle': 'Comparación de Rendimiento (Precisión)',
                'description': 'Este gráfico compara la precisión de los diferentes modelos, ofreciendo una visión rápida de cuál modelo tuvo el mejor desempeño en términos de clasificación correcta.'
            },
            'training_time_comparison.png': {
                'subtitle': 'Comparación del Tiempo de Entrenamiento',
                'description': 'Este gráfico ilustra el tiempo que cada modelo tardó en entrenarse, lo cual es crucial para evaluar la eficiencia computacional de cada arquitectura.'
            },
            'error_histogram.png': {
                'subtitle': 'Histograma de Errores',
                'description': 'Este histograma visualiza la distribución de los errores de predicción, ayudando a identificar si los modelos tienden a cometer errores en ciertas clases o con cierta magnitud.'
            },
            'prediction_correlation_matrix.png': {
                'subtitle': 'Matriz de Correlación de Predicciones',
                'description': 'Esta matriz muestra la correlación entre las predicciones de los diferentes modelos, indicando qué tan a menudo los modelos están de acuerdo o en desacuerdo en sus clasificaciones.'
            },
            'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png': {
                'subtitle': 'Correlación de Predicciones: ResNet18 vs DenseNet121',
                'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
            },
            'prediction_correlation_matrix_ResNet18_vs_ResNet50.png': {
                'subtitle': 'Correlación de Predicciones: ResNet18 vs ResNet50',
                'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y ResNet50, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
            },
            'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png': {
                'subtitle': 'Correlación de Predicciones: ResNet50 vs DenseNet121',
                'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet50 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
            }
        }

        results_dir = 'results'
        
        # Filter out specific confusion matrices and the last 3 images
        all_image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        specific_cm_files = [files['confusion_matrix'] for files in model_results_info.values()]
        
        # Images to exclude based on user's request
        images_to_exclude = [
            'training_validation_plot_DenseNet121.png',
            'training_validation_plot_ResNet18.png',
            'training_validation_plot_ResNet50.png'
        ]

        other_image_files = [f for f in all_image_files if f not in specific_cm_files and f not in images_to_exclude]
        
        # Sort the remaining files to ensure consistent order
        sorted_other_image_files = sorted(other_image_files)

        # Display images in two columns
        cols = st.columns(2)
        for i, img_file in enumerate(sorted_other_image_files):
            with cols[i % 2]: # Alternate between column 0 and column 1
                details = image_details.get(img_file, {'subtitle': img_file.replace('_', ' ').replace('.png', ''), 'description': 'Descripción no disponible.'})
                st.markdown(f"#### {details['subtitle']}")
                st.markdown(f"_{details['description']}_")
                # Removed 'height' as it's not a valid argument for st.image and adjusted width
                st.image(os.path.join(results_dir, img_file), use_container_width=True)
                # Removed duplicate st.image call


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

    elif page == 'Generar Reporte PDF':
        st.header('Generación de Reporte PDF 📄')
        st.markdown('Haz clic en el botón para generar un reporte PDF con todos los resultados estadísticos y gráficos.')

        if st.button('Generar Reporte PDF'):
            st.info("Generando reporte PDF... Esto puede tomar un momento.")
            # Call the PDF generation function here
            generate_pdf_report()
            st.success("Reporte PDF generado exitosamente. Puedes encontrarlo en la carpeta 'results'.")
            st.download_button(
                label="Descargar Reporte PDF",
                data=open("results/reporte_enfermedades_patata.pdf", "rb").read(),
                file_name="reporte_enfermedades_patata.pdf",
                mime="application/pdf"
            )

# Define a custom page template for headers/footers
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 0.75 * inch, f"Página {doc.page}")
    canvas.restoreState()

def generate_pdf_report():
    doc = SimpleDocTemplate("results/reporte_enfermedades_patata.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom style for centered title
    styles.add(ParagraphStyle(name='TitleStyle', alignment=TA_CENTER, fontSize=36, spaceAfter=24, leading=40))
    styles.add(ParagraphStyle(name='SubtitleStyle', alignment=TA_CENTER, fontSize=20, spaceAfter=18, leading=24))
    styles.add(ParagraphStyle(name='DateStyle', alignment=TA_CENTER, fontSize=12, spaceAfter=72))
    # Modify the existing 'Italic' style instead of adding a new one
    styles['Italic'].fontName = 'Helvetica-Oblique'
    styles['Italic'].fontSize = 10
    styles['Italic'].textColor = colors.grey
    if 'Code' not in styles:
        styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=8, leading=9, backColor=colors.lightgrey, borderPadding=6))

    elements = []

    # Title Page
    elements.append(Spacer(1, 2 * inch)) # Top margin
    elements.append(Paragraph("Reporte de Análisis de Enfermedades", styles['TitleStyle']))
    elements.append(Paragraph("de la Hoja de Patata", styles['TitleStyle']))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Análisis Estadístico y Visualización de Modelos de Aprendizaje Profundo", styles['SubtitleStyle']))
    elements.append(Spacer(1, 1.0 * inch))
    elements.append(Paragraph(f"Fecha de Generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['DateStyle']))
    elements.append(PageBreak())

    # Section: Statistical Analysis
    elements.append(Paragraph("1. Análisis Estadístico de Modelos", styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))

    results_dir = 'results'
    model_names = ['ResNet18', 'ResNet50', 'DenseNet121']

    # Confusion Matrices, Classification Reports, and MCC
    elements.append(Paragraph("1.1. Matrices de Confusión, Informes de Clasificación y MCC", styles['h2']))
    
    # Confusion Matrices, Classification Reports, and MCC (Sequential Display)
    elements.append(Paragraph("1.1. Matrices de Confusión, Informes de Clasificación y MCC", styles['h2']))
    for model_name in model_names:
        elements.append(Paragraph(f"Resultados para {model_name}:", styles['h3']))
        
        # Confusion Matrix Image
        cm_path = os.path.join(results_dir, f'confusion_matrix_{model_name.lower()}.png')
        if os.path.exists(cm_path):
            elements.append(RLImage(cm_path, width=4*inch, height=3.2*inch)) # Reverted to larger size for single column
            elements.append(Paragraph(f"Matriz de Confusión para {model_name}", styles['Italic']))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Classification Report
        cr_path = os.path.join(results_dir, f'classification_report_{model_name.lower()}.txt')
        if os.path.exists(cr_path):
            with open(cr_path, 'r') as f:
                report_text = f.read()
            elements.append(Paragraph(f"Informe de Clasificación para {model_name}:", styles['h4']))
            elements.append(Preformatted(report_text, styles['Code']))
            elements.append(Spacer(1, 0.1 * inch))

        # Matthews Correlation Coefficient
        eval_json_path = os.path.join(results_dir, f'evaluation_results_potato_leaf_disease_model_{model_name.lower()}.json')
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r') as f:
                eval_data = json.load(f)
                if 'matthews_corrcoef' in eval_data:
                    elements.append(Paragraph(f"Coeficiente de Matthews (MCC) para {model_name}:", styles['h4']))
                    elements.append(Paragraph(f"MCC: {eval_data['matthews_corrcoef']:.4f}", styles['Normal']))
                    elements.append(Spacer(1, 0.1 * inch))
        elements.append(PageBreak()) # Add page break after each model's results

    # McNemar's Test Results
    elements.append(Paragraph("1.2. Prueba de McNemar para Comparación de Modelos", styles['h2']))
    elements.append(Spacer(1, 0.1 * inch))
    mcnemar_results_path = os.path.join(results_dir, 'mcnemar_test_results.json')
    if os.path.exists(mcnemar_results_path):
        with open(mcnemar_results_path, 'r') as f:
            mcnemar_results = json.load(f)
        
        data = [['Modelo 1', 'Modelo 2', 'Estadístico Chi-cuadrado', 'Valor p']] # Removed 'Conclusión'
        for result in mcnemar_results:
            model1_name = result['model1']
            model2_name = result['model2']
            if 'statistic' in result['results']:
                statistic = f"{result['results']['statistic']:.4f}"
                pvalue = f"{result['results']['pvalue']:.4f}"
                # conclusion = result['results']['conclusion'] # Removed conclusion from here
            else:
                statistic = "N/A"
                pvalue = "N/A"
                # conclusion = f"Error: {result['results']['error']}" # Removed conclusion from here
            data.append([model1_name, model2_name, statistic, pvalue]) # Removed conclusion from append

        # Calculate column widths dynamically or set them explicitly
        # For letter size (8.5 x 11 inches), usable width is about 6.5 inches (8.5 - 1 inch margins on each side)
        # 6.5 inches = 6.5 * 72 points = 468 points
        col_widths = [1.5 * inch, 1.5 * inch, 1.5 * inch, 2.0 * inch] # Adjusted widths for 4 columns

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), # Vertically align content in cells
        ])
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        # Add conclusions below the table
        elements.append(Paragraph("Conclusiones de la Prueba de McNemar:", styles['h3']))
        for result in mcnemar_results:
            if 'statistic' in result['results']:
                conclusion_text = f"• Comparación {result['model1']} vs {result['model2']}: {result['results']['conclusion']}"
            else:
                conclusion_text = f"• Comparación {result['model1']} vs {result['model2']}: Error al realizar la prueba: {result['results']['error']}"
            elements.append(Paragraph(conclusion_text, styles['Normal']))
            elements.append(Spacer(1, 0.1 * inch))
    else:
        elements.append(Paragraph("No se encontraron resultados de la Prueba de McNemar. Ejecuta 'scripts/plot_results.py' para generarlos.", styles['Normal']))
    elements.append(PageBreak())

    # Section: Performance Plots
    elements.append(Paragraph("2. Gráficos de Rendimiento", styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))

    image_details = {
        'accuracy_per_model.png': {
            'subtitle': 'Precisión por Modelo',
            'description': 'Este gráfico muestra la precisión general de cada modelo (ResNet18, ResNet50, DenseNet121) en el conjunto de datos de validación.'
        },
        'performance_comparison_accuracy.png': {
            'subtitle': 'Comparación de Rendimiento (Precisión)',
            'description': 'Este gráfico compara la precisión de los diferentes modelos, ofreciendo una visión rápida de cuál modelo tuvo el mejor desempeño en términos de clasificación correcta.'
        },
        'training_time_comparison.png': {
            'subtitle': 'Comparación del Tiempo de Entrenamiento',
            'description': 'Este gráfico ilustra el tiempo que cada modelo tardó en entrenarse, lo cual es crucial para evaluar la eficiencia computacional de cada arquitectura.'
        },
        'error_histogram.png': {
            'subtitle': 'Histograma de Errores',
            'description': 'Este histograma visualiza la distribución de los errores de predicción, ayudando a identificar si los modelos tienden a cometer errores en ciertas clases o con cierta magnitud.'
        },
        'prediction_correlation_matrix.png': {
            'subtitle': 'Matriz de Correlación de Predicciones',
            'description': 'Esta matriz muestra la correlación entre las predicciones de los diferentes modelos, indicando qué tan a menudo los modelos están de acuerdo o en desacuerdo en sus clasificaciones.'
        },
        'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png': {
            'subtitle': 'Correlación de Predicciones: ResNet18 vs DenseNet121',
            'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
        },
        'prediction_correlation_matrix_ResNet18_vs_ResNet50.png': {
            'subtitle': 'Correlación de Predicciones: ResNet18 vs ResNet50',
            'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y ResNet50, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
        },
        'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png': {
            'subtitle': 'Correlación de Predicciones: ResNet50 vs DenseNet121',
            'description': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet50 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.'
        }
    }

    all_image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
    specific_cm_files = [f'confusion_matrix_{name.lower()}.png' for name in model_names]
    images_to_exclude = [
        'training_validation_plot_DenseNet121.png',
        'training_validation_plot_ResNet18.png',
        'training_validation_plot_ResNet50.png'
    ]
    other_image_files = [f for f in all_image_files if f not in specific_cm_files and f not in images_to_exclude]
    sorted_other_image_files = sorted(other_image_files)

    for img_file in sorted_other_image_files:
        details = image_details.get(img_file, {'subtitle': img_file.replace('_', ' ').replace('.png', ''), 'description': 'Descripción no disponible.'})
        elements.append(Paragraph(details['subtitle'], styles['h3']))
        elements.append(Paragraph(details['description'], styles['Italic']))
        elements.append(RLImage(os.path.join(results_dir, img_file), width=6*inch, height=4*inch))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(PageBreak()) # Start new page for each major plot

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)

if __name__ == '__main__':
    main()
