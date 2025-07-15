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
from googletrans import Translator

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Preformatted, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib import colors

# Configuración multilenguaje
translator = Translator()
LANGUAGES = {
    'English': 'en',
    'Español': 'es',
    'Français': 'fr',
    'Deutsch': 'de'
}

# Cargar traducciones
@st.cache_data(ttl=3600) # Cache translations for 1 hour
def load_translations(lang_code):
    translations = {
        'app_title': {
            'en': 'Potato Leaf Disease Detector',
            'es': 'Detector de Enfermedades de la Hoja de Patata',
            'fr': 'Détecteur de Maladies des Feuilles de Pomme de Terre',
            'de': 'Kartoffelblattkrankheitsdetektor'
        },
        'navigation_title': {
            'en': 'Navigation',
            'es': 'Navegación',
            'fr': 'Navigation',
            'de': 'Navigation'
        },
        'page_home_models': {
            'en': 'Home & Models',
            'es': 'Inicio y Modelos',
            'fr': 'Accueil et Modèles',
            'de': 'Startseite & Modelle'
        },
        'page_evaluation_results': {
            'en': 'Evaluation Results',
            'es': 'Resultados de Evaluación',
            'fr': 'Résultats d\'Évaluation',
            'de': 'Evaluierungsergebnisse'
        },
        'page_image_prediction': {
            'en': 'Image Prediction',
            'es': 'Predicción de Imagen',
            'fr': 'Prédiction d\'Image',
            'de': 'Bildvorhersage'
        },
        'page_generate_report': {
            'en': 'Generate PDF Report',
            'es': 'Generar Reporte PDF',
            'fr': 'Générer un Rapport PDF',
            'de': 'PDF-Bericht generieren'
        },
        'page_disease_info': {
            'en': 'Disease Information',
            'es': 'Información de Enfermedades',
            'fr': 'Informations sur les Maladies',
            'de': 'Krankheitsinformationen'
        },
        'page_recommendations': {
            'en': 'Recommendations',
            'es': 'Recomendaciones',
            'fr': 'Recommandations',
            'de': 'Empfehlungen'
        },
        'welcome_header': {
            'en': 'Welcome 👋 and General Information 📚',
            'es': 'Bienvenido 👋 y Información General 📚',
            'fr': 'Bienvenue 👋 et Informations Générales 📚',
            'de': 'Willkommen 👋 y Allgemeine Informationen 📚'
        },
        'welcome_text': {
            'en': 'This application uses deep learning models to detect diseases in potato leaves. Simply upload an image of a potato leaf and our models will help you identify possible diseases. Our goal is to provide a useful tool for farmers and agricultural enthusiasts to quickly identify crop problems and take timely action.',
            'es': 'Esta aplicación utiliza modelos de aprendizaje profundo para detectar enfermedades en las hojas de patata. Simplemente sube una imagen de una hoja de patata y nuestros modelos te ayudarán a identificar posibles enfermedades. Nuestro objetivo es proporcionar una herramienta útil para agricultores y entusiastas de la agricultura para identificar rápidamente problemas en los cultivos y tomar medidas oportunas.',
            'fr': 'Cette application utilise des modèles d\'apprentissage profond pour détecter les maladies des feuilles de pomme de terre. Téléchargez simplement une image d\'une feuille de pomme de terre et nos modèles vous aideront à identifier les maladies possibles. Notre objectif est de fournir un outil utile aux agriculteurs et aux passionnés d\'agriculture pour identifier rapidement les problèmes de culture et prendre des mesures opportunes.',
            'de': 'Diese Anwendung verwendet Deep-Learning-Modelle, um Krankheiten an Kartoffelblättern zu erkennen. Laden Sie einfach ein Bild eines Kartoffelblatts hoch, und unsere Modelle helfen Ihnen, mögliche Krankheiten zu identifizieren. Unser Ziel ist es, Landwirten und Agrarbegeisterten ein nützliches Werkzeug zur schnellen Erkennung von Pflanzenproblemen und zur rechtzeitigen Maßnahmen zu bieten.'
        },
        'explore_info': {
            'en': 'Explore the sections below to learn more about the dataset and models.',
            'es': 'Explora las secciones a continuación para conocer más sobre el dataset y los modelos.',
            'fr': 'Explorez les sections ci-dessous pour en savoir plus sur l\'ensemble de données et les modèles.',
            'de': 'Erkunden Sie die folgenden Abschnitte, um mehr über den Datensatz und die Modelle zu erfahren.'
        },
        'dataset_info_subheader': {
            'en': 'Dataset Information',
            'es': 'Información del Dataset',
            'fr': 'Informations sur l\'Ensemble de Données',
            'de': 'Datensatzinformationen'
        },
        'dataset_description': {
            'en': 'The models were trained with the **"Potato Leaf Disease Dataset in Uncontrolled Environment"** https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset .This dataset contains images of potato leaves classified into different disease categories and healthy leaves. Below is the distribution of classes in the dataset:',
            'es': 'Los modelos fueron entrenados con el **"Potato Leaf Disease Dataset in Uncontrolled Environment"** https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset .Este dataset contiene imágenes de hojas de patata clasificadas en diferentes categorías de enfermedades y hojas sanas. A continuación, se muestra la distribución de las clases en el dataset:',
            'fr': 'Les modèles ont été entraînés avec le **"Potato Leaf Disease Dataset in Uncontrolled Environment"** https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset .Cet ensemble de données contient des images de feuilles de pomme de terre classées en différentes catégories de maladies et en feuilles saines. Voici la distribution des classes dans l\'ensemble de données :',
            'de': 'Die Modelle wurden mit dem **"Potato Leaf Disease Dataset in Uncontrolled Environment"** https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset trainiert. Dieser Datensatz enthält Bilder von Kartoffelblättern, die in verschiedene Krankheitskategorien und gesunde Blätter eingeteilt sind. Unten ist die Verteilung der Klassen im Datensatz dargestellt:'
        },
        'class_distribution': {
            'en': 'Class Distribution:',
            'es': 'Distribución de Clases:',
            'fr': 'Distribution des Classes :',
            'de': 'Klassenverteilung:'
        },
        'neural_network_models_subheader': {
            'en': 'Neural Network Models Used',
            'es': 'Modelos de Red Neuronal Utilizados',
            'fr': 'Modèles de Réseaux Neuronaux Utilisés',
            'de': 'Verwendete Neuronale Netzwerkmodelle'
        },
        'models_description': {
            'en': 'We have employed three state-of-the-art convolutional neural network (CNN) architectures for potato leaf disease classification:',
            'es': 'Hemos empleado tres arquitecturas de redes neuronales convolucionales (CNN) de última generación para la clasificación de enfermedades en hojas de patata:',
            'fr': 'Nous avons utilizado tres arquitecturas de redes neuronales convolucionales (CNN) de última generación para la clasificación de enfermedades en hojas de patata:',
            'de': 'Wir haben drei hochmoderne Faltungsnetzwerk-Architekturen (CNN) zur Klassifizierung von Kartoffelblattkrankheiten eingesetzt:'
        },
        'details_for': {
            'en': 'Details for',
            'es': 'Detalles de',
            'fr': 'Détails pour',
            'de': 'Details für'
        },
        'resnet18_desc': {
            'en': '**ResNet18** is a lighter version of the ResNet (Residual Networks) family. These networks introduce skip connections that allow the gradient to flow directly through multiple layers, which helps train very deep networks without vanishing gradient problems. ResNet18 is known for its efficiency and good performance.',
            'es': '**ResNet18** es una versión más ligera de la familia ResNet (Residual Networks). Estas redes introducen conexiones de salto (skip connections) que permiten que el gradiente fluya directamente a través de múltiples capas, lo que ayuda a entrenar redes muy profundas sin problemas de desvanecimiento del gradiente. ResNet18 es conocida por su eficiencia y buen rendimiento.',
            'fr': '**ResNet18** est une version plus légère de la famille ResNet (Residual Networks). Ces réseaux introduisent des connexions de saut (skip connections) qui permettent au gradient de circuler directement à travers plusieurs couches, ce qui aide à entraîner des réseaux très profonds sans problèmes de disparition du gradient. ResNet18 est connu pour son efficacité et ses bonnes performances.',
            'de': '**ResNet18** ist eine leichtere Version der ResNet-Familie (Residual Networks). Diese Netzwerke führen Sprungverbindungen (Skip Connections) ein, die es dem Gradienten ermöglichen, direkt durch mehrere Schichten zu fließen, was hilft, sehr tiefe Netzwerke ohne Probleme des verschwindenden Gradienten zu trainieren. ResNet18 ist bekannt für seine Effizienz und gute Leistung.'
        },
        'resnet50_desc': {
            'en': '**ResNet50** is a deeper version of ResNet than ResNet18. It uses more layers and residual blocks, which allows it to learn more complex features and often achieve higher accuracy in image classification tasks, albeit with a higher computational cost.',
            'es': '**ResNet50** es una versión más profunda de ResNet que ResNet18. Utiliza más capas y bloques residuales, lo que le permite aprender características más complejas y, a menudo, lograr una mayor precisión en tareas de clasificación de imágenes, aunque con un mayor costo computacional.',
            'fr': '**ResNet50** est una versión más profunda de ResNet que ResNet18. Il utilise plus de couches et de blocs résiduels, ce qui lui permet d\'apprendre des caractéristiques plus complexes et d\'atteindre souvent une plus grande précision dans les tâches de classification d\'images, bien qu\'avec un coût de calcul plus élevé.',
            'de': '**ResNet50** ist eine tiefere Version von ResNet als ResNet18. Es verwendet mehr Schichten und Residualblöcke, wodurch es komplexere Merkmale lernen und oft eine höhere Genauigkeit bei Bildklassifizierungsaufgaben erzielen kann, wenn auch mit höheren Rechenkosten.'
        },
        'densenet121_desc': {
            'en': '**DenseNet121** (Densely Connected Convolutional Networks) is an architecture that connects each layer with all subsequent layers in a "feed-forward" fashion. This means that the input to each layer consists of the output of all previous layers, which encourages feature reuse and reduces the number of parameters, improving information and gradient propagation.',
            'es': '**DenseNet121** (Densely Connected Convolutional Networks) es una arquitectura que conecta cada capa con todas las capas posteriores en una moda de "alimentación hacia adelante". Esto significa que la entrada de cada capa consiste en la salida de todas las capas anteriores, lo que fomenta la reutilización de características y reduce el número de parámetros, mejorando la propagación de la información y el gradiente.',
            'fr': '**DenseNet121** (Densely Connected Convolutional Networks) es una arquitectura que conecta cada capa con todas las capas posteriores en una moda de "alimentación hacia adelante". Esto significa que la entrada de cada capa consiste en la salida de todas las capas anteriores, lo que fomenta la reutilización de características y reduce el número de parámetros, mejorando la propagación de la información y el gradiente.',
            'de': '**DenseNet121** (Densely Connected Convolutional Networks) ist eine Architektur, die jede Schicht mit allen nachfolgenden Schichten in einer "Feed-Forward"-Weise verbindet. Dies bedeutet, dass die Eingabe jeder Schicht aus der Ausgabe aller vorherigen Schichten besteht, was die Wiederverwendung von Merkmalen fördert und die Anzahl der Parameter reduziert, wodurch die Informations- und Gradientenpropagation verbessert wird.'
        },
        'evaluation_results_header': {
            'en': 'Model Evaluation Results 📊',
            'es': 'Resultados de Evaluación de los Modelos 📊',
            'fr': 'Résultats d\'Évaluation des Modèles 📊',
            'de': 'Modell-Evaluierungsergebnisse 📊'
        },
        'evaluation_results_description': {
            'en': 'Here you can see the metrics and performance graphs of the trained models.',
            'es': 'Aquí puedes ver las métricas y gráficos de rendimiento de los modelos entrenados.',
            'fr': 'Ici, vous pouvez voir les métriques et les graphiques de performance des modèles entraînés.',
            'de': 'Hier sehen Sie die Metriken und Leistungsdiagramme der trainierten Modelle.'
        },
        'detailed_results_for': {
            'en': 'Detailed Results for',
            'es': 'Resultados Detallados para',
            'fr': 'Resultados Détaillés para',
            'de': 'Detaillierte Ergebnisse für'
        },
        'confusion_matrix_for': {
            'en': 'Confusion Matrix for',
            'es': 'Matriz de Confusión para',
            'fr': 'Matrice de Confusion para',
            'de': 'Konfusionsmatrix für'
        },
        'no_confusion_matrix': {
            'en': 'No confusion matrix found for',
            'es': 'No se encontró la matriz de confusión para',
            'fr': 'Aucune matriz de confusión encontrada para',
            'de': 'Keine Konfusionsmatrix gefunden für'
        },
        'classification_report_for': {
            'en': 'Classification Report for',
            'es': 'Informe de Clasificación para',
            'fr': 'Rapport de Classification para',
            'de': 'Klassifizierungsbericht für'
        },
        'no_classification_report': {
            'en': 'No classification report found for',
            'es': 'No se encontró el informe de clasificación para',
            'fr': 'Aucun rapport de clasificación encontrado para',
            'de': 'Kein Klassifizierungsbericht gefunden für'
        },
        'matthews_corrcoef_for': {
            'en': 'Matthews Correlation Coefficient (MCC) for',
            'es': 'Coeficiente de Matthews (MCC) para',
            'fr': 'Coefficient de Corrélation de Matthews (MCC) para',
            'de': 'Matthews Korrelationskoeffizient (MCC) für'
        },
        'no_matthews_corrcoef': {
            'en': 'No Matthews Coefficient found for',
            'es': 'No se encontró el Coeficiente de Matthews para',
            'fr': 'Aucun Coeficiente de Matthews encontrado para',
            'de': 'Kein Matthews-Koeffizient gefunden für'
        },
        'no_evaluation_results': {
            'en': 'No evaluation results found for',
            'es': 'No se encontraron resultados de evaluación para',
            'fr': 'Aucun resultado de evaluación encontrado para',
            'de': 'Keine Evaluierungsergebnisse gefunden für'
        },
        'mcnemar_test_subheader': {
            'en': 'McNemar\'s Test Results for Model Comparison:',
            'es': 'Resultados de la Prueba de McNemar para Comparación de Modelos:',
            'fr': 'Résultats du Test de McNemar para la Comparación de Modèles :',
            'de': 'McNemar-Testergebnisse für den Modellvergleich:'
        },
        'comparison': {
            'en': 'Comparison:',
            'es': 'Comparación:',
            'fr': 'Comparaison :',
            'de': 'Vergleich:'
        },
        'chi_squared_statistic': {
            'en': 'Chi-squared Statistic:',
            'es': 'Estadístico Chi-cuadrado:',
            'fr': 'Statistique du Chi-carré :',
            'de': 'Chi-Quadrat-Statistik:'
        },
        'p_value': {
            'en': 'p-value:',
            'es': 'Valor p:',
            'fr': 'Valeur p :',
            'de': 'p-Wert:'
        },
        'conclusion': {
            'en': 'Conclusion:',
            'es': 'Conclusión:',
            'fr': 'Conclusión:',
            'de': 'Fazit:'
        },
        'error_performing_test': {
            'en': 'Error performing test:',
            'es': 'Error al realizar la prueba:',
            'fr': 'Erreur lors de l\'exécution du test :',
            'de': 'Fehler beim Ausführen des Tests:'
        },
        'no_mcnemar_results': {
            'en': 'No McNemar\'s Test results found. Run \'scripts/plot_results.py\' to generate them.',
            'es': 'No se encontraron resultados de la Prueba de McNemar. Ejecuta \'scripts/plot_results.py\' para generarlos.',
            'fr': 'Aucun resultado del Test de McNemar encontrado. Exécutez \'scripts/plot_results.py\' para les générer.',
            'de': 'Keine McNemar-Testergebnisse gefunden. Führen Sie \'scripts/plot_results.py\' aus, um sie zu generieren.'
        },
        'other_performance_graphs': {
            'en': 'Other Performance Graphs:',
            'es': 'Otros Gráficos de Rendimiento:',
            'fr': 'Autres Graphiques de Performance :',
            'de': 'Weitere Leistungsdiagramme:'
        },
        'accuracy_per_model_subtitle': {
            'en': 'Accuracy per Model',
            'es': 'Precisión por Modelo',
            'fr': 'Précision por Modèle',
            'de': 'Genauigkeit pro Modell'
        },
        'accuracy_per_model_desc': {
            'en': 'This graph shows the overall accuracy of each model (ResNet18, ResNet50, DenseNet121) on the validation dataset.',
            'es': 'Este gráfico muestra la precisión general de cada modelo (ResNet18, ResNet50, DenseNet121) en el conjunto de datos de validación.',
            'fr': 'Ce gráfico muestra la precisión global de cada modelo (ResNet18, ResNet50, DenseNet121) sobre el conjunto de datos de validación.',
            'de': 'Dieses Diagramm zeigt die Gesamtgenauigkeit jedes Modells (ResNet18, ResNet50, DenseNet121) auf dem Validierungsdatensatz.'
        },
        'performance_comparison_accuracy_subtitle': {
            'en': 'Performance Comparison (Accuracy)',
            'es': 'Comparación de Rendimiento (Precisión)',
            'fr': 'Comparaison des Performances (Précisión)',
            'de': 'Leistungsvergleich (Genauigkeit)'
        },
        'performance_comparison_accuracy_desc': {
            'en': 'This graph compares the accuracy of the different models, offering a quick overview of which model performed best in terms of correct classification.',
            'es': 'Este gráfico compara la precisión de los diferentes modelos, ofreciendo una visión rápida de cuál modelo tuvo el mejor desempeño en términos de clasificación correcta.',
            'fr': 'Ce gráfico compara la precisión de los diferentes modelos, ofreciendo un aperçu rápido de cuál modelo tuvo el mejor desempeño en términos de clasificación correcta.',
            'de': 'Dieses Diagramm vergleicht die Genauigkeit der verschiedenen Modelle y ofrece un rápido resumen de cuál modelo tuvo el mejor desempeño en términos de clasificación correcta.'
        },
        'training_time_comparison_subtitle': {
            'en': 'Training Time Comparison',
            'es': 'Comparación del Tiempo de Entrenamiento',
            'fr': 'Comparaison du Temps d\'Entraînement',
            'de': 'Vergleich der Trainingszeit'
        },
        'training_time_comparison_desc': {
            'en': 'This graph illustrates the time each model took to train, which is crucial for evaluating the computational efficiency of each architecture.',
            'es': 'Este gráfico ilustra el tiempo que cada modelo tardó en entrenarse, lo cual es crucial para evaluar la eficiencia computacional de cada arquitectura.',
            'fr': 'Ce gráfico ilustra el tiempo que cada modelo tardó en entrenarse, lo cual es crucial para evaluar la eficiencia computacional de cada arquitectura.',
            'de': 'Dieses Diagramm veranschaulicht die Trainingszeit jedes Modells, was entscheidend für die Bewertung der Recheneffizienz jeder Architektur ist.'
        },
        'error_histogram_subtitle': {
            'en': 'Error Histogram',
            'es': 'Histograma de Errores',
            'fr': 'Histogramme des Erreurs',
            'de': 'Fehlerhistogramm'
        },
        'error_histogram_desc': {
            'en': 'This histogram visualizes the distribution of prediction errors, helping to identify if models tend to make errors in certain classes or with a certain magnitude.',
            'es': 'Este histograma visualiza la distribución de los errores de predicción, ayudando a identificar si los modelos tienden a cometer errores en ciertas clases o con cierta magnitud.',
            'fr': 'Cet histogramme visualiza la distribución de los errores de predicción, ayudando a identificar si los modelos tienden a cometer errores en ciertas clases o con cierta magnitud.',
            'de': 'Dieses Histogramm visualisiert die Verteilung der Vorhersagefehler und hilft dabei, zu identifizieren, ob Modelle dazu neigen, Fehler in bestimmten Klassen oder mit einer bestimmten Größe zu machen.'
        },
        'prediction_correlation_matrix_subtitle': {
            'en': 'Prediction Correlation Matrix',
            'es': 'Matriz de Correlación de Predicciones',
            'fr': 'Matrice de Corrélation des Prédictions',
            'de': 'Vorhersage-Korrelationsmatrix'
        },
        'prediction_correlation_matrix_desc': {
            'en': 'This matrix shows the correlation between the predictions of the different models, indicating how often the models agree or disagree in their classifications.',
            'es': 'Esta matriz muestra la correlación entre las predicciones de los diferentes modelos, indicando qué tan a menudo los modelos están de acuerdo o en desacuerdo en sus clasificaciones.',
            'fr': 'Cette matriz muestra la correlación entre las predicciones de los diferentes modelos, indicando à quelle fréquence les modèles sont d\'accord o en desacuerdo dans leurs classifications.',
            'de': 'Diese Matrix zeigt die Korrelation zwischen den Vorhersagen der verschiedenen Modelle und gibt an, wie oft die Modelle in ihren Klassifizierungen übereinstimmen oder nicht übereinstimmen.'
        },
        'prediction_correlation_matrix_resnet18_densenet121_subtitle': {
            'en': 'Prediction Correlation: ResNet18 vs DenseNet121',
            'es': 'Correlación de Predicciones: ResNet18 vs DenseNet121',
            'fr': 'Corrélation des Prédictions : ResNet18 vs DenseNet121',
            'de': 'Vorhersage-Korrelation: ResNet18 vs DenseNet121'
        },
        'prediction_correlation_matrix_resnet18_densenet121_desc': {
            'en': 'This specific matrix details the correlation between the predictions of ResNet18 and DenseNet121, revealing patterns of agreement and disagreement between these two models.',
            'es': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'fr': 'Cette matriz específica detalla la correlación entre las predicciones de ResNet18 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'de': 'Diese spezifische Matrix zeigt die Korrelation zwischen den Vorhersagen von ResNet18 und DenseNet121 und enthüllt Muster der Übereinstimmung und Nichtübereinstimmung zwischen diesen beiden Modellen.'
        },
        'prediction_correlation_matrix_resnet18_resnet50_subtitle': {
            'en': 'Prediction Correlation: ResNet18 vs ResNet50',
            'es': 'Correlación de Predicciones: ResNet18 vs ResNet50',
            'fr': 'Corrélation des Prédictions : ResNet18 vs ResNet50',
            'de': 'Vorhersage-Korrelation: ResNet18 vs ResNet50'
        },
        'prediction_correlation_matrix_resnet18_resnet50_desc': {
            'en': 'This specific matrix details the correlation between the predictions of ResNet18 and ResNet50, revealing patterns of agreement and disagreement between these two models.',
            'es': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet18 y ResNet50, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'fr': 'Cette matriz específica detalla la correlación entre las predicciones de ResNet18 y ResNet50, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'de': 'Diese específica Matrix zeigt die Korrelation zwischen den Vorhersagen von ResNet18 und ResNet50 und enthüllt Muster der Übereinstimmung und Nichtübereinstimmung zwischen diesen beiden Modellen.'
        },
        'prediction_correlation_matrix_resnet50_densenet121_subtitle': {
            'en': 'Prediction Correlation: ResNet50 vs DenseNet121',
            'es': 'Correlación de Predicciones: ResNet50 vs DenseNet121',
            'fr': 'Corrélation des Prédictions : ResNet50 vs DenseNet121',
            'de': 'Vorhersage-Korrelation: ResNet50 vs DenseNet121'
        },
        'prediction_correlation_matrix_resnet50_densenet121_desc': {
            'en': 'This specific matrix details the correlation between the predictions of ResNet50 and DenseNet121, revealing patterns of agreement and disagreement between these two models.',
            'es': 'Esta matriz específica detalla la correlación entre las predicciones de ResNet50 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'fr': 'Cette matriz específica detalla la correlación entre las predicciones de ResNet50 y DenseNet121, revelando patrones de acuerdo y desacuerdo entre estos dos modelos.',
            'de': 'Diese específica Matrix zeigt die Korrelation zwischen den Vorhersagen von ResNet50 und DenseNet121 und enthüllt Muster der Übereinstimmung und Nichtübereinstimmung zwischen diesen beiden Modellen.'
        },
        'prediction_header': {
            'en': 'Potato Leaf Disease Prediction 🔍',
            'es': 'Predicción de Enfermedades de la Hoja de Patata 🔍',
            'fr': 'Prédiction des Maladies des Feuilles de Pomme de Terre 🔍',
            'de': 'Kartoffelblattkrankheitsvorhersage 🔍'
        },
        'prediction_description': {
            'en': 'Upload an image of a potato leaf for the models to make a prediction.',
            'es': 'Sube una imagen de una hoja de patata para que los modelos realicen una predicción.',
            'fr': 'Téléchargez una imagen de una hoja de pomme de terre para que los modelos realicen una predicción.',
            'de': 'Laden Sie ein Bild eines Kartoffelblatts hoch, damit die Modelle eine Vorhersage treffen können.'
        },
        'choose_image': {
            'en': 'Choose an image',
            'es': 'Elige una imagen',
            'fr': 'Choisissez una imagen',
            'de': 'Wählen Sie ein Bild'
        },
        'uploaded_image': {
            'en': 'Uploaded Image',
            'es': 'Imagen Cargada',
            'fr': 'Imagen Téléchargée',
            'de': 'Hochgeladenes Bild'
        },
        'perform_prediction': {
            'en': 'Perform Prediction',
            'es': 'Realizar Predicción',
            'fr': 'Effectuer la Prédiction',
            'de': 'Vorhersage durchführen'
        },
        'prediction_results_subheader': {
            'en': 'Prediction Results:',
            'es': 'Resultados de la Predicción:',
            'fr': 'Resultados de la Predicción :',
            'de': 'Vorhersageergebnisse:'
        },
        'detected_disease': {
            'en': 'Detected Disease:',
            'es': 'Enfermedad Detectada:',
            'fr': 'Maladie Détectée :',
            'de': 'Erkannte Krankheit:'
        },
        'confidence': {
            'en': 'Confidence:',
            'es': 'Confianza:',
            'fr': 'Confiance :',
            'de': 'Konfidenz:'
        },
        'class_probabilities': {
            'en': 'Class Probabilities',
            'es': 'Probabilidades de Clase',
            'fr': 'Probabilités de Classe',
            'de': 'Klassenwahrscheinlichkeiten'
        },
        'generate_report_header': {
            'en': 'PDF Report Generation 📄',
            'es': 'Generación de Reporte PDF 📄',
            'fr': 'Génération de Rapport PDF 📄',
            'de': 'PDF-Berichtserstellung 📄'
        },
        'generate_report_description': {
            'en': 'Click the button to generate a PDF report with all statistical results and graphs.',
            'es': 'Haz clic en el botón para generar un reporte PDF con todos los resultados estadísticos y gráficos.',
            'fr': 'Cliquez sur le botón para generar un rapport PDF con todos los resultados estadísticos y gráficos.',
            'de': 'Klicken Sie auf die Schaltfläche, um einen PDF-Bericht mit allen statistischen Ergebnissen und Diagrammen zu erstellen.'
        },
        'generate_report_button': {
            'en': 'Generate PDF Report',
            'es': 'Generar Reporte PDF',
            'fr': 'Générer un Rapport PDF',
            'de': 'PDF-Bericht generieren'
        },
        'generating_report': {
            'en': 'Generando reporte PDF... Esto puede tomar un momento.',
            'es': 'Generando reporte PDF... Esto puede tomar un momento.',
            'fr': 'Génération du rapport PDF... Cela peut prendre un moment.',
            'de': 'PDF-Bericht wird generiert... Dies kann einen Moment dauern.'
        },
        'report_generated_successfully': {
            'en': 'PDF report generated successfully.',
            'es': 'Reporte PDF generado exitosamente.',
            'fr': 'Rapport PDF généré avec succès.',
            'de': 'PDF-Bericht erfolgreich generiert.'
        },
        'download_report': {
            'en': 'Download PDF Report',
            'es': 'Descargar Reporte PDF',
            'fr': 'Descargar Reporte PDF',
            'de': 'PDF-Bericht herunterladen'
        },
        'report_title_pdf': {
            'en': 'Disease Analysis Report',
            'es': 'Reporte de Análisis de Enfermedades',
            'fr': 'Rapport d\'Analyse des Maladies',
            'de': 'Krankheitsanalysebericht'
        },
        'report_subtitle_pdf': {
            'en': 'Statistical Analysis and Visualization of Deep Learning Models',
            'es': 'Análisis Estadístico y Visualización de Modelos de Aprendizaje Profundo',
            'fr': 'Analyse Statistique et Visualisation des Modèles d\'Apprentissage Profond',
            'de': 'Statistische Analyse und Visualisierung von Deep-Learning-Modellen'
        },
        'generation_date_pdf': {
            'en': 'Generation Date:',
            'es': 'Fecha de Generación:',
            'fr': 'Date de Génération :',
            'de': 'Generierungsdatum:'
        },
        'statistical_analysis_section': {
            'en': '1. Statistical Analysis of Models',
            'es': '1. Análisis Estadístico de Modelos',
            'fr': '1. Analyse Statistique des Modèles',
            'de': '1. Statistische Analyse von Modellen'
        },
        'confusion_reports_mcc_section': {
            'en': '1.1. Confusion Matrices, Classification Reports, and MCC',
            'es': '1.1. Matrices de Confusión, Informes de Clasificación y MCC',
            'fr': '1.1. Matrices de Confusion, Rapports de Classification et MCC',
            'de': '1.1. Konfusionsmatrizen, Klassifizierungsberichte und MCC'
        },
        'results_for': {
            'en': 'Results for',
            'es': 'Resultados para',
            'fr': 'Resultados para',
            'de': 'Ergebnisse für'
        },
        'confusion_matrix_caption': {
            'en': 'Confusion Matrix for',
            'es': 'Matriz de Confusión para',
            'fr': 'Matrice de Confusion para',
            'de': 'Konfusionsmatrix für'
        },
        'classification_report_caption': {
            'en': 'Classification Report for',
            'es': 'Informe de Clasificación para',
            'fr': 'Rapport de Classification para',
            'de': 'Klassifizierungsbericht für'
        },
        'mcc_caption': {
            'en': 'Matthews Correlation Coefficient (MCC) for',
            'es': 'Coeficiente de Matthews (MCC) para',
            'fr': 'Coefficient de Corrélation de Matthews (MCC) para',
            'de': 'Matthews Korrelationskoeffizient (MCC) für'
        },
        'mcnemar_test_section': {
            'en': '1.2. McNemar\'s Test for Model Comparison',
            'es': '1.2. Prueba de McNemar para Comparación de Modelos',
            'fr': '1.2. Test de McNemar para la Comparación de Modèles',
            'de': '1.2. McNemar-Test für den Modellvergleich'
        },
        'model1': {
            'en': 'Model 1',
            'es': 'Modelo 1',
            'fr': 'Modèle 1',
            'de': 'Modell 1'
        },
        'model2': {
            'en': 'Model 2',
            'es': 'Modelo 2',
            'fr': 'Modèle 2',
            'de': 'Modell 2'
        },
        'chi_squared_statistic_table': {
            'en': 'Chi-squared Statistic',
            'es': 'Estadístico Chi-cuadrado',
            'fr': 'Statistique du Chi-carré',
            'de': 'Chi-Quadrat-Statistik'
        },
        'p_value_table': {
            'en': 'p-value',
            'es': 'Valor p',
            'fr': 'Valor p',
            'de': 'p-Wert'
        },
        'conclusions_mcnemar': {
            'en': 'Conclusions from McNemar\'s Test:',
            'es': 'Conclusiones de la Prueba de McNemar:',
            'fr': 'Conclusiones del Test de McNemar :',
            'de': 'Schlussfolgerungen aus dem McNemar-Test:'
        },
        'no_mcnemar_results_pdf': {
            'en': 'No McNemar\'s Test results found. Run \'scripts/plot_results.py\' to generate them.',
            'es': 'No se encontraron resultados de la Prueba de McNemar. Ejecuta \'scripts/plot_results.py\' para generarlos.',
            'fr': 'Aucun resultado del Test de McNemar encontrado. Exécutez \'scripts/plot_results.py\' para les générer.',
            'de': 'Keine McNemar-Testergebnisse gefunden. Führen Sie \'scripts/plot_results.py\' aus, um sie zu generieren.'
        },
        'performance_plots_section': {
            'en': '2. Performance Plots',
            'es': '2. Gráficos de Rendimiento',
            'fr': '2. Graphiques de Performance',
            'de': '2. Leistungsdiagramme'
        },
        'description_not_available': {
            'en': 'Description not available.',
            'es': 'Descripción no disponible.',
            'fr': 'Description non disponible.',
            'de': 'Beschreibung nicht disponible.'
        },
        'disease_info_header': {
            'en': 'Potato Leaf Disease Information 🌿',
            'es': 'Información de Enfermedades de la Hoja de Patata 🌿',
            'fr': 'Informations sur les Maladies des Feuilles de Pomme de Terre 🌿',
            'de': 'Informationen zu Kartoffelblattkrankheiten 🌿'
        },
        'disease_info_intro': {
            'en': 'Here you can find detailed information about the different potato leaf diseases that our models can detect, along with their symptoms and characteristics.',
            'es': 'Aquí puedes encontrar información detallada sobre las diferentes enfermedades de la hoja de patata que nuestros modelos pueden detectar, junto con sus síntomas y características.',
            'fr': 'Ici, vous trouverez des informations détaillées sur les différentes enfermedades de las hojas de pomme de terre que nos modèles peuvent détecter, ainsi que leurs symptômes et caractéristiques.',
            'de': 'Hier finden Sie detaillierte Informationen zu den verschiedenen Kartoffelblattkrankheiten, die unsere Modelle erkennen können, zusammen mit ihren Symptomen und Merkmalen.'
        },
        'bacteria_name': {
            'en': 'Bacteria',
            'es': 'Bacteria',
            'fr': 'Bactéries',
            'de': 'Bakterien'
        },
        'bacteria_desc': {
            'en': 'Bacterial diseases often cause water-soaked lesions, wilting, and soft rot. They can spread rapidly in warm, humid conditions.',
            'es': 'Las enfermedades bacterianas a menudo causan lesiones empapadas de agua, marchitamiento y pudrición blanda. Pueden propagarse rápidamente en condiciones cálidas y húmedas.',
            'fr': 'Les maladies bactériennes provoquent souvent des lésions gorgées d\'eau, le flétrissement y la pourriture molle. Elles pueden propagarse rápidamente en condiciones cálidas y húmedas.',
            'de': 'Bakterielle Krankheiten verursachen oft wassergetränkte Läsionen, Welke und Weichfäule. Sie können sich bei warmen, feuchten Bedingungen schnell ausbreiten.'
        },
        'fungi_name': {
            'en': 'Fungi',
            'es': 'Hongos',
            'fr': 'Champignons',
            'de': 'Pilze'
        },
        'fungi_desc': {
            'en': 'Fungal diseases typically appear as spots, mold, or powdery growth on leaves. They thrive in moist environments and can severely impact yield.',
            'es': 'Las enfermedades fúngicas suelen aparecer como manchas, moho o crecimiento polvoriento en las hojas. Prosperan en ambientes húmedos y pueden afectar gravemente el rendimiento.',
            'fr': 'Las enfermedades fúngicas suelen aparecer como manchas, moho o crecimiento polvoriento en las hojas. Prosperan en ambientes húmedos y pueden afectar gravemente el rendimiento.',
            'de': 'Pilzkrankheiten treten typischerweise als Flecken, Schimmel oder pulverförmiges Wachstum auf den Blättern auf. Sie gedeihen in feuchten Umgebungen und können den Ertrag stark beeinträchtigen.'
        },
        'healthy_name': {
            'en': 'Healthy',
            'es': 'Sana',
            'fr': 'Sain',
            'de': 'Gesund'
        },
        'healthy_desc': {
            'en': 'Healthy potato leaves are vibrant green, firm, and show no signs of discoloration, spots, or deformities. They indicate a thriving plant.',
            'es': 'Las hojas de patata sanas son de color verde vibrante, firmes y no muestran signos de decoloración, manchas o deformidades. Indican una planta próspera.',
            'fr': 'Las hojas de patata sanas son de color verde vibrante, firmes y no muestran signos de decoloración, manchas o deformidades. Elles indican una planta florissante.',
            'de': 'Gesunde Kartoffelblätter sind leuchtend grün, fest und zeigen keine Anzeichen von Verfärbungen, Flecken oder Deformationen. Sie weisen auf eine gedeihende Pflanze hin.'
        },
        'nematode_name': {
            'en': 'Nematode',
            'es': 'Nematodo',
            'fr': 'Nématodes',
            'de': 'Nematoden'
        },
        'nematode_desc': {
            'en': 'Nematode damage can cause stunted growth, yellowing, and wilting, often mimicking nutrient deficiencies. They attack roots, affecting water and nutrient uptake.',
            'es': 'El daño por nematodos puede causar un crecimiento atrofiado, amarillamiento y marchitamiento, a menudo imitando deficiencias de nutrientes. Atacan las raíces, afectando la absorción de agua y nutrientes.',
            'fr': 'El daño por nematodos puede causar un crecimiento atrofiado, amarillamiento y marchitamiento, a menudo imitando deficiencias de nutrientes. Atacan las raíces, afectando la absorción de agua y nutrientes.',
            'de': 'Nematodenschäden können zu Wachstumsstörungen, Vergilbung und Welke führen, oft Nährstoffmangel imitierend. Sie greifen die Wurzeln an und beeinträchtigen die Wasser- und Nährstoffaufnahme.'
        },
        'pest_name': {
            'en': 'Pest',
            'es': 'Plaga',
            'fr': 'Ravageur',
            'de': 'Schädling'
        },
        'pest_desc': {
            'en': 'Pest infestations lead to visible damage like chewed leaves, holes, or presence of insects. Common potato pests include aphids, leafhoppers, and potato beetles.',
            'es': 'Las infestaciones de plagas provocan daños visibles como hojas masticadas, agujeros o presencia de insectos. Las plagas comunes de la patata incluyen pulgones, cicadélidos y escarabajos de la patata.',
            'fr': 'Las infestaciones de plagas provocan daños visibles como hojas masticadas, agujeros o presencia de insectos. Las plagas comunes de la patata incluyen pulgones, cicadélidos y escarabajos de la patata.',
            'de': 'Schädlingsbefall führt zu sichtbaren Schäden wie angefressenen Blättern, Löchern oder dem Vorhandensein von Insekten. Häufige Kartoffelschädlinge sind Blattläuse, Zikaden y Kartoffelkäfer.'
        },
        'phytophthora_name': {
            'en': 'Phytophthora',
            'es': 'Phytophthora',
            'fr': 'Phytophthora',
            'de': 'Phytophthora'
        },
        'phytophthora_desc': {
            'en': 'Phytophthora infestans causes late blight, characterized by dark, water-soaked lesions on leaves that rapidly expand, often with a fuzzy white mold on the underside.',
            'es': 'Phytophthora infestans causa el tizón tardío, caracterizado por lesiones oscuras y empapadas de agua en las hojas que se expanden rápidamente, a menudo con un moho blanco y algodonoso en el envés.',
            'fr': 'Phytophthora infestans causa el tizón tardío, caracterizado por lesiones oscuras y empapadas de agua en las hojas que se expanden rápidamente, a menudo con un moho blanco y algodonoso en el envés.',
            'de': 'Phytophthora infestans verursacht die Kraut- und Knollenfäule, gekennzeichnet durch dunkle, wassergetränkte Läsionen auf den Blättern, die sich schnell ausbreiten, oft mit einem flaumigen weißen Schimmel auf der Unterseite.'
        },
        'virus_name': {
            'en': 'Virus',
            'es': 'Virus',
            'fr': 'Virus',
            'de': 'Virus'
        },
        'virus_desc': {
            'en': 'Viral diseases often result in mosaic patterns, curling, or stunted growth. They are typically spread by insect vectors and can significantly reduce plant vigor.',
            'es': 'Las enfermedades virales a menudo resultan en patrones de mosaico, rizado o crecimiento atrofiado. Típicamente se propagan por vectores de insectos y pueden reducir significativamente el vigor de la planta.',
            'fr': 'Las enfermedades virales a menudo resultan en patrones de mosaico, rizado o crecimiento atrofiado. Típicamente se propagan por vectores de insectos y pueden reducir significativamente el vigor de la planta.',
            'de': 'Virale Krankheiten führen oft zu Mosaikmustern, Kräuselungen oder Wachstumsstörungen. Sie werden typischerweise durch Insektenvektoren verbreitet und können die Pflanzenvitalität erheblich reduzieren.'
        },
        'recommendations_header': {
            'en': 'Recommendations for Disease Management 💡',
            'es': 'Recomendaciones para el Manejo de Enfermedades 💡',
            'fr': 'Recommandations para la Gestion des Maladies 💡',
            'de': 'Empfehlungen zur Krankheitsbekämpfung 💡'
        },
        'recommendations_intro': {
            'en': 'Based on the detected disease, here are some general recommendations for management and prevention. Always consult with a local agricultural expert for specific advice tailored to your region and conditions.',
            'es': 'Basado en la enfermedad detectada, aquí hay algunas recomendaciones generales para el manejo y la prevención. Siempre consulta con un experto agrícola local para obtener asesoramiento específico adaptado a tu región y condiciones.',
            'fr': 'Basado en la enfermedad detectada, aquí hay algunas recomendaciones generales para el manejo y la prevención. Siempre consulta con un experto agrícola local para obtener asesoramiento específico adaptado a tu región y condiciones.',
            'de': 'Basierend auf der erkannten Krankheit finden Sie hier einige allgemeine Empfehlungen zur Bekämpfung und Prävention. Konsultieren Sie immer einen lokalen Landwirtschaftsexperten für spezifische Ratschläge, die auf Ihre Region und Bedingungen zugeschnitten sind.'
        },
        'general_recommendations': {
            'en': 'General Recommendations:',
            'es': 'Recomendaciones Generales:',
            'fr': 'Recommandations Générales :',
            'de': 'Allgemeine Empfehlungen:'
        },
        'general_recommendation_1': {
            'en': 'Crop Rotation: Rotate potato crops with non-host plants to break disease cycles.',
            'es': 'Rotación de Cultivos: Rota los cultivos de patata con plantas no hospedadoras para romper los ciclos de enfermedades.',
            'fr': 'Rotation des Cultivos : Rota los cultivos de patata con plantas no hospedadoras para romper los ciclos de enfermedades.',
            'de': 'Fruchtwechsel: Wechseln Sie Kartoffelpflanzen mit Nicht-Wirtspflanzen ab, um Krankheitszyklen zu unterbrechen.'
        },
        'general_recommendation_2': {
            'en': 'Sanitation: Remove and destroy infected plant debris to reduce inoculum sources.',
            'es': 'Saneamiento: Elimina y destruye los restos de plantas infectadas para reducir las fuentes de inóculo.',
            'fr': 'Saneamiento: Elimina y destruye los restos de plantas infectadas para reducir las fuentes de inóculo.',
            'de': 'Sanierung: Entfernen und zerstören Sie infizierte Pflanzenreste, um Inokulumquellen zu reduzieren.'
        },
        'general_recommendation_3': {
            'en': 'Resistant Varieties: Plant potato varieties known to be resistant to common diseases in your area.',
            'es': 'Variedades Resistentes: Planta variedades de patata conocidas por ser resistentes a enfermedades comunes en tu área.',
            'fr': 'Variedades Resistentes: Planta variedades de patata conocidas por ser resistentes a enfermedades comunes en tu área.',
            'de': 'Resistente Sorten: Pflanzen Sie Kartoffelsorten, die bekanntermaßen resistent gegen häufige Krankheiten in Ihrer Region sind.'
        },
        'general_recommendation_4': {
            'en': 'Proper Irrigation: Avoid overhead irrigation that keeps leaves wet for prolonged periods, which favors fungal and bacterial growth.',
            'es': 'Riego Adecuado: Evita el riego por aspersión que mantiene las hojas mojadas durante períodos prolongados, lo que favorece el crecimiento de hongos y bacterias.',
            'fr': 'Riego Adecuado: Evita el riego por aspersión que mantiene las hojas mojadas durante períodos prolongados, lo que favorece el crecimiento de hongos y bacterias.',
            'de': 'Richtige Bewässerung: Vermeiden Sie Überkopfbewässerung, die Blätter über längere Zeit feucht hält, was das Wachstum von Pilzen und Bakterien begünstigt.'
        },
        'general_recommendation_5': {
            'en': 'Nutrient Management: Ensure balanced fertilization to promote strong plant health and resilience.',
            'es': 'Manejo de Nutrientes: Asegura una fertilización equilibrada para promover una fuerte salud y resistencia de la planta.',
            'fr': 'Manejo de Nutrientes: Asegura una fertilización equilibrada para promover una fuerte salud y resistencia de la planta.',
            'de': 'Nährstoffmanagement: Sorgen Sie für eine ausgewogene Düngung, um eine starke Pflanzengesundheit und Widerstandsfähigkeit zu fördern.'
        },
        'specific_recommendations': {
            'en': 'Specific Recommendations for',
            'es': 'Recomendaciones Específicas para',
            'fr': 'Recomendaciones Específicas para',
            'de': 'Spezifische Empfehlungen für'
        },
        'bacteria_reco': {
            'en': 'Use copper-based bactericides. Improve drainage and avoid working in wet fields to prevent spread.',
            'es': 'Usa bactericidas a base de cobre. Mejora el drenaje y evita trabajar en campos húmedos para prevenir la propagación.',
            'fr': 'Usa bactericidas a base de cobre. Mejora el drenaje y evita trabajar en campos húmedos para prevenir la propagación.',
            'de': 'Verwenden Sie kupferbasierte Bakterizide. Verbessern Sie die Drainage und vermeiden Sie das Arbeiten auf nassen Feldern, um die Ausbreitung zu verhindern.'
        },
        'fungi_reco': {
            'en': 'Apply fungicides as per label instructions. Ensure good air circulation around plants.',
            'es': 'Aplica fungicidas según las instrucciones de la etiqueta. Asegura una buena circulación de aire alrededor de las plantas.',
            'fr': 'Aplica fungicidas según las instrucciones de la etiqueta. Asegura una buena circulación de aire alrededor de las plantas.',
            'de': 'Wenden Sie Fungizide gemäß den Anweisungen auf dem Etikett an. Sorgen Sie für eine gute Luftzirkulation um die Pflanzen.'
        },
        'nematode_reco': {
            'en': 'Consider soil solarization or nematicides. Plant cover crops that suppress nematodes.',
            'es': 'Considera la solarización del suelo o nematicidas. Planta cultivos de cobertura que supriman los nematodos.',
            'fr': 'Considera la solarización del suelo o nematicidas. Planta cultivos de cobertura que supriman los nematodos.',
            'de': 'Erwägen Sie Bodensolarisation oder Nematizide. Pflanzen Sie Zwischenfrüchte, die Nematoden unterdrücken.'
        },
        'pest_reco': {
            'en': 'Implement integrated pest management (IPM) strategies, including biological controls and appropriate insecticides.',
            'es': 'Implementa estrategias de manejo integrado de plagas (MIP), incluyendo controles biológicos e insecticidas apropiados.',
            'fr': 'Implementa estrategias de manejo integrado de plagas (MIP), incluyendo controles biológicos e insecticidas apropiados.',
            'de': 'Implementieren Sie integrierte Schädlingsbekämpfungsstrategien (IPM), einschließlich biologischer Kontrollen und geeigneter Insektizide.'
        },
        'phytophthora_reco': {
            'en': 'Apply fungicides specifically for late blight. Improve air circulation and avoid overhead irrigation. Remove and destroy infected plants and debris promptly.',
            'es': 'Aplica fungicidas específicos para el tizón tardío. Mejora la circulación del aire y evita el riego por aspersión. Elimina y destruye las plantas y restos infectados rápidamente.',
            'fr': 'Appliquez des fungicidas específicos para el mildiou. Améliorez la circulación de l\'air y evite el riego por aspersión. Retire y destruya rápidamente las plantas y los restos infectados.',
            'de': 'Wenden Sie Fungizide speziell gegen Kraut- und Knollenfäule an. Verbessern Sie die Luftzirkulation und vermeiden Sie Überkopfbewässerung. Entfernen und vernichten Sie infizierte Pflanzen und Pflanzenreste umgehend.'
        },
        'virus_reco': {
            'en': 'Control insect vectors (e.g., aphids) that spread viruses. Remove and destroy infected plants immediately.',
            'es': 'Controla los insectos vectores (ej. pulgones) que propagan virus. Elimina y destruye las plantas infectadas inmediatamente.',
            'fr': 'Controla los insectos vectores (ej. pulgones) que propagan virus. Elimina y destruye las plantas infectadas inmediatamente.',
            'de': 'Kontrollieren Sie Insektenvektoren (z. B. Blattläuse), die Viren verbreiten. Entfernen und zerstören Sie infizierte Pflanzen sofort.'
        },
        'healthy_reco': {
            'en': 'Maintain good agricultural practices, including proper fertilization, irrigation, and pest control, to ensure continued plant health.',
            'es': 'Mantén buenas prácticas agrícolas, incluyendo fertilización adecuada, riego y control de plagas, para asegurar la salud continua de la planta.',
            'fr': 'Maintenez de bonnes pratiques agricoles, y compris une fertilisation, une irrigation et un control des ravageurs apropiados, para asegurar la salud continua de las plantas.',
            'de': 'Pflegen Sie gute landwirtschaftliche Praktiken, einschließlich ordnungsgemäßer Düngung, Bewässerung und Schädlingsbekämpfung, um die anhaltende Pflanzengesundheit zu gewährleisten.'
        },
        'no_specific_reco': {
            'en': 'No specific recommendations available for this class.',
            'es': 'No hay recomendaciones específicas disponibles para esta clase.',
            'fr': 'Aucune recomendación específica disponible para esta clase.',
            'de': 'Keine spezifischen Empfehlungen für diese Klasse verfügbar.'
        },
        'prediction_history_header': {
            'en': 'Prediction History 📜',
            'es': 'Historial de Predicciones 📜',
            'fr': 'Historique des Prédictions 📜',
            'de': 'Vorhersageverlauf 📜'
        },
        'no_predictions_yet': {
            'en': 'No predictions yet. Upload an image to start!',
            'es': 'Aún no hay predicciones. ¡Sube una imagen para empezar!',
            'fr': 'Aucune prédiction pour l\'instant. Téléchargez una imagen para comenzar !',
            'de': 'Noch keine Vorhersagen. Laden Sie ein Bild hoch, um zu beginnen!'
        },
        'clear_history': {
            'en': 'Clear History',
            'es': 'Borrar Historial',
            'fr': 'Effacer l\'Historique',
            'de': 'Verlauf löschen'
        },
        'download_current_prediction': {
            'en': 'Download Current Prediction',
            'es': 'Descargar Predicción Actual',
            'fr': 'Télécharger la Prédiction Actuelle',
            'de': 'Aktuelle Vorhersage herunterladen'
        },
        'specific_recommendations_title': {
            'en': 'Specific Recommendations',
            'es': 'Recomendaciones Específicas',
            'fr': 'Recommandations Spécifiques',
            'de': 'Spezifische Empfehlungen'
        },
        'training_validation_plot_desc': {
            'en': 'This plot shows the training and validation accuracy and loss curves for the model, indicating its learning progress and potential overfitting.',
            'es': 'Este gráfico muestra las curvas de precisión y pérdida de entrenamiento y validación para el modelo, indicando su progreso de aprendizaje y posible sobreajuste.',
            'fr': 'Ce gráfico muestra las curvas de precisión y pérdida de entrenamiento y validación para el modelo, indicando su progreso de aprendizaje y posible sobreajuste.',
            'de': 'Dieses Diagramm zeigt die Trainings- und Validierungsgenauigkeits- und Verlustkurven für das Modell, die den Lernfortschritt und das potenzielle Overfitting anzeigen.'
        },
        'page_text': {
            'en': 'Page',
            'es': 'Página',
            'fr': 'Page',
            'de': 'Seite'
        },
        'probability': {
            'en': 'Probability',
            'es': 'Probabilidad',
            'fr': 'Probabilité',
            'de': 'Wahrscheinlichkeit'
        },
        'app_title_suffix_pdf': {
            'en': 'of Potato Leaf',
            'es': 'de la Hoja de Patata',
            'fr': 'de la Feuille de Pomme de Terre',
            'de': 'des Kartoffelblattes'
        },
        'model_features_section': {
            'en': '3. Model Features',
            'es': '3. Características de los Modelos',
            'fr': '3. Caractéristiques des Modèles',
            'de': '3. Modellmerkmale'
        },
        'model_comparison_section': {
            'en': '5. Model Comparison',
            'es': '5. Comparación de Modelos',
            'fr': '5. Comparaison des Modèles',
            'de': '5. Modellvergleich'
        },
        'statistical_analysis_plots_section': {
            'en': '6. Statistical Analysis and Performance Plots',
            'es': '6. Análisis Estadístico y Gráficos de Rendimiento',
            'fr': '6. Analyse Statistique et Graphiques de Performance',
            'de': '6. Statistische Analyse und Gráficos de Rendimiento'
        },
        'last_prediction_pdf': {
            'en': '1. Last Model Prediction',
            'es': '1. Última Predicción del Modelo',
            'fr': '1. Dernière Prédiction du Modèle',
            'de': '1. Letzte Modellvorhersage'
        },
        'model_used_pdf': {
            'en': 'Model Used:',
            'es': 'Modelo Utilizado:',
            'fr': 'Modèle Utilisé :',
            'de': 'Verwendetes Modelo:'
        },
        'detected_disease_pdf': {
            'en': 'Detected Disease:',
            'es': 'Enfermedad Detectada:',
            'fr': 'Maladie Détectée :',
            'de': 'Erkannte Krankheit:'
        },
        'confidence_pdf': {
            'en': 'Confidence:',
            'es': 'Confianza:',
            'fr': 'Confiance :',
            'de': 'Konfidenz:'
        },
        'class_pdf': {
            'en': 'Class',
            'es': 'Clase',
            'fr': 'Classe',
            'de': 'Klasse'
        },
        'probability_pdf': {
            'en': 'Probability',
            'es': 'Probabilidad',
            'fr': 'Probabilité',
            'de': 'Wahrscheinlichkeit'
        },
        'specific_recommendation_for_pdf': {
            'en': 'Specific Recommendation for',
            'es': 'Recomendación Específica para',
            'fr': 'Recommandation Spécifique pour',
            'de': 'Spezifische Empfehlung für'
        },
        'class_names_map': {
            'en': {
                'Bacteria': 'Bacteria',
                'Fungi': 'Fungi',
                'Healthy': 'Healthy',
                'Nematode': 'Nematode',
                'Pest': 'Pest',
                'Phytophthora': 'Phytophthora',
                'Virus': 'Virus'
            },
            'es': {
                'Bacteria': 'Bacteria',
                'Fungi': 'Hongos',
                'Healthy': 'Sana',
                'Nematode': 'Nematodo',
                'Pest': 'Plaga',
                'Phytophthora': 'Phytophthora',
                'Virus': 'Virus'
            },
            'fr': {
                'Bacteria': 'Bactéries',
                'Fungi': 'Champignons',
                'Healthy': 'Sain',
                'Nematode': 'Nématodes',
                'Pest': 'Ravageur',
                'Phytophthora': 'Phytophthora',
                'Virus': 'Virus'
            },
            'de': {
                'Bacteria': 'Bakterien',
                'Fungi': 'Pilze',
                'Healthy': 'Gesund',
                'Nematode': 'Nematoden',
                'Pest': 'Schädling',
                'Phytophthora': 'Phytophthora',
                'Virus': 'Virus'
            }
        },
        'roc_curve_title': {
            'en': 'Receiver Operating Characteristic (ROC) Curve for',
            'es': 'Curva Característica Operativa del Receptor (ROC) para',
            'fr': 'Courbe Caractéristique de Fonctionnement du Récepteur (ROC) para',
            'de': 'Receiver Operating Characteristic (ROC) Kurve für'
        },
        'false_positive_rate': {
            'en': 'False Positive Rate',
            'es': 'Tasa de Falsos Positivos',
            'fr': 'Taux de Faux Positifs',
            'de': 'Falsch-Positiv-Rate'
        },
        'true_positive_rate': {
            'en': 'True Positive Rate',
            'es': 'Tasa de Verdaderos Positivos',
            'fr': 'Taux de Vrais Positifs',
            'de': 'Wahre-Positiv-Rate'
        },
        'auc_score': {
            'en': 'AUC Score',
            'es': 'Puntuación AUC',
            'fr': 'Score AUC',
            'de': 'AUC-Wert'
        }
    }
    # Add lang_code to the translations dictionary
    translations_for_lang = {k: v[lang_code] for k, v in translations.items()}
    translations_for_lang['lang_code'] = lang_code
    return translations_for_lang

# Cargar traducciones desde el archivo JSON
@st.cache_data(ttl=3600) # Cache translations for 1 hour
def load_translations_from_json(lang_code):
    translations_path = "translations.json"
    if not os.path.exists(translations_path):
        st.error(f"Error: No se encontró el archivo de traducciones en: {translations_path}")
        return {} # Retornar un diccionario vacío o manejar el error apropiadamente
    
    with open(translations_path, 'r', encoding='utf-8') as f:
        all_translations = json.load(f)
    
    translations_for_lang = {k: v[lang_code] for k, v in all_translations.items()}
    translations_for_lang['lang_code'] = lang_code
    return translations_for_lang

CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus']
CLASS_NAMES_ENGLISH = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus'] # Mantener los nombres en inglés para el mapeo

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
    # Inicializar session_state para el historial de predicciones si no existe
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    st.set_page_config(page_title='Detector de Enfermedades de la Hoja de Patata', layout='wide')
    
    # Configuración multilenguaje en la barra lateral
    st.sidebar.title("Idioma/Language")
    
    # Obtener el idioma actual de la sesión o establecer 'es' por defecto
    if 'current_lang_code' not in st.session_state:
        st.session_state.current_lang_code = 'es' # Idioma por defecto

    # Crear el selectbox para el idioma
    lang_name = st.sidebar.selectbox("", list(LANGUAGES.keys()), 
                                     index=list(LANGUAGES.values()).index(st.session_state.current_lang_code),
                                     label_visibility='hidden',
                                     key='lang_selector')

    # Detectar si el idioma ha cambiado
    new_lang_code = LANGUAGES[lang_name]
    if new_lang_code != st.session_state.current_lang_code:
        st.session_state.current_lang_code = new_lang_code
        # Regenerar los gráficos en el nuevo idioma
        try:
            import subprocess
            python_executable = sys.executable
            command = [python_executable, "scripts/plot_results.py", "--lang", st.session_state.current_lang_code]
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Salida de plot_results.py (stdout):", process.stdout)
            if process.stderr:
                print("Salida de plot_results.py (stderr):", process.stderr)
            st.rerun() # Recargar la página para mostrar los gráficos actualizados
        except subprocess.CalledProcessError as e:
            st.error(f"Error al generar los gráficos en el idioma seleccionado: {e.stderr}")
            print(f"Error al generar los gráficos: {e.stderr}")
        except FileNotFoundError:
            st.error("Error: No se encontró el script 'scripts/plot_results.py'. Asegúrate de que el archivo existe.")
        
    t = load_translations_from_json(st.session_state.current_lang_code)

    st.title(t['app_title'])

    st.sidebar.title(t['navigation_title'])
    page = st.sidebar.radio(
        t['navigation_title'],
        [t['page_home_models'], t['page_evaluation_results'], t['page_image_prediction'], t['page_generate_report'], t['page_disease_info'], t['prediction_history_header']],
        label_visibility='hidden' # Ocultar la etiqueta para evitar la advertencia
    )

    models_dict = load_models()

    if page == t['page_home_models']:
        st.header(t['welcome_header'])
        st.markdown(t['welcome_text'])
        st.info(t['explore_info'])

        st.markdown("---") # Separador

        col_home1, col_home2 = st.columns(2)

        with col_home1:
            st.subheader(t['dataset_info_subheader'])
            st.markdown(t['dataset_description'])
            dataset_info = {
                "Bacteria": 569,
                "Fungi": 748,
                "Healthy": 201,
                "Nematode": 68,
                "Pest": 611,
                "Phytophthora": 347,
                "Virus": 532
            }
            
            st.write(t['class_distribution'])
            
            # Traducir las claves del diccionario dataset_info para la visualización JSON
            translated_dataset_info = {t['class_names_map'][key]: value for key, value in dataset_info.items()}
            st.json(translated_dataset_info)
            
            # Traducir las etiquetas para el gráfico de pastel
            labels = [t['class_names_map'][label] for label in dataset_info.keys()]
            sizes = list(dataset_info.values())
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax1.axis('equal')
            st.pyplot(fig1)
            plt.close(fig1)

        with col_home2:
            st.subheader(t['neural_network_models_subheader'])
            st.markdown(t['models_description'])

            model_names = ['ResNet18', 'ResNet50', 'DenseNet121']
            model_descriptions = {
                'ResNet18': t['resnet18_desc'],
                'ResNet50': t['resnet50_desc'],
                'DenseNet121': t['densenet121_desc']
            }

            for model_name in model_names:
                with st.expander(f"{t['details_for']} {model_name}"):
                    st.markdown(model_descriptions[model_name])

    elif page == t['page_evaluation_results']:
        st.header(t['evaluation_results_header'])
        st.markdown(t['evaluation_results_description'])

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
            with st.expander(f"{t['detailed_results_for']} {model_name}"):
                col_cm, col_cr = st.columns(2)

                with col_cm:
                    st.subheader(f"{t['confusion_matrix_for']} {model_name}:")
                    cm_path = os.path.join(results_dir, files['confusion_matrix'])
                    if os.path.exists(cm_path):
                        st.image(cm_path, caption=f"{t['confusion_matrix_for']} {model_name}")
                    else:
                        st.info(f"{t['no_confusion_matrix']} {model_name}.")

                with col_cr:
                    st.subheader(f"{t['classification_report_for']} {model_name}:")
                    cr_path = os.path.join(results_dir, files['classification_report'])
                    if os.path.exists(cr_path):
                        with open(cr_path, 'r') as f:
                            st.text(f.read())
                    else:
                        st.info(f"{t['no_classification_report']} {model_name}.")
                    
                    # Mostrar el Coeficiente de Correlación de Matthews (MCC)
                    eval_json_path = os.path.join(results_dir, f'evaluation_results_potato_leaf_disease_model_{model_name.lower()}.json')
                    if os.path.exists(eval_json_path):
                        with open(eval_json_path, 'r') as f:
                            eval_data = json.load(f)
                            if 'matthews_corrcoef' in eval_data:
                                st.subheader(f"{t['matthews_corrcoef_for']} {model_name}:")
                                st.write(f"**MCC:** {eval_data['matthews_corrcoef']:.4f}")
                            else:
                                st.info(f"{t['no_matthews_corrcoef']} {model_name}.")
                    else:
                        st.info(f"{t['no_evaluation_results']} {model_name}.")

        st.markdown("---") # Separador

        st.subheader(t['mcnemar_test_subheader'])
        mcnemar_results_path = os.path.join(results_dir, 'mcnemar_test_results.json')
        if os.path.exists(mcnemar_results_path):
            with open(mcnemar_results_path, 'r') as f:
                mcnemar_results = json.load(f)
            
            for result in mcnemar_results:
                st.markdown(f"##### {t['comparison']} {result['model1']} vs {result['model2']}")
                if 'statistic' in result['results']:
                    st.write(f"{t['chi_squared_statistic']}: {result['results']['statistic']:.4f}")
                    st.write(f"{t['p_value']}: {result['results']['pvalue']:.4f}")
                    translated_mcnemar_conclusion = translator.translate(result['results']['conclusion'], dest=t['lang_code']).text
                    st.write(f"{t['conclusion']}: {translated_mcnemar_conclusion}")
                else:
                    st.write(f"{t['error_performing_test']}: {result['results']['error']}")
                st.markdown("---")
        else:
            st.info(t['no_mcnemar_results'])

        st.markdown("---") # Separador

        st.subheader(t['other_performance_graphs'])
        
        # Define image details with subtitles and descriptions
        image_details = {
            'accuracy_per_model.png': {
                'subtitle': t['accuracy_per_model_subtitle'],
                'description': t['accuracy_per_model_desc']
            },
            'performance_comparison_accuracy.png': {
                'subtitle': t['performance_comparison_accuracy_subtitle'],
                'description': t['performance_comparison_accuracy_desc']
            },
            'training_time_comparison.png': {
                'subtitle': t['training_time_comparison_subtitle'],
                'description': t['training_time_comparison_desc']
            },
            'error_histogram.png': {
                'subtitle': t['error_histogram_subtitle'],
                'description': t['error_histogram_desc']
            },
            'prediction_correlation_matrix.png': {
                'subtitle': t['prediction_correlation_matrix_subtitle'],
                'description': t['prediction_correlation_matrix_desc']
            },
            'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png': {
                'subtitle': t['prediction_correlation_matrix_resnet18_densenet121_subtitle'],
                'description': t['prediction_correlation_matrix_resnet18_densenet121_desc']
            },
            'prediction_correlation_matrix_ResNet18_vs_ResNet50.png': {
                'subtitle': t['prediction_correlation_matrix_resnet18_resnet50_subtitle'],
                'description': t['prediction_correlation_matrix_resnet18_resnet50_desc']
            },
            'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png': {
                'subtitle': t['prediction_correlation_matrix_resnet50_densenet121_subtitle'],
                'description': t['prediction_correlation_matrix_resnet50_densenet121_desc']
            },
            'training_validation_plot_DenseNet121.png': {
                'subtitle': f"{t['results_for']} DenseNet121",
                'description': f"{t['training_validation_plot_desc']} DenseNet121"
            },
            'training_validation_plot_ResNet18.png': {
                'subtitle': f"{t['results_for']} ResNet18",
                'description': f"{t['training_validation_plot_desc']} ResNet18"
            },
            'training_validation_plot_ResNet50.png': {
                'subtitle': f"{t['results_for']} ResNet50",
                'description': f"{t['training_validation_plot_desc']} ResNet50"
            }
        }

        results_dir = 'results'
        
        # Filter out specific confusion matrices, training plots, and prediction correlation matrix
        all_image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        specific_cm_files = [files['confusion_matrix'] for files in model_results_info.values()]
        
        # Exclude training validation plots and prediction correlation matrices from display in the app
        plots_to_exclude = [
            'training_validation_plot_DenseNet121.png',
            'training_validation_plot_ResNet18.png',
            'training_validation_plot_ResNet50.png',
            'prediction_correlation_matrix.png', # Eliminar esta imagen
            'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png',
            'prediction_correlation_matrix_ResNet18_vs_ResNet50.png',
            'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png'
        ]

        other_image_files = [f for f in all_image_files if f not in specific_cm_files and f not in plots_to_exclude]
        sorted_other_image_files = sorted(other_image_files)

        # Display images in two columns
        cols = st.columns(2)
        for i, img_file in enumerate(sorted_other_image_files):
            with cols[i % 2]: # Alternate between column 0 and column 1
                details = image_details.get(img_file, {'subtitle': img_file.replace('_', ' ').replace('.png', ''), 'description': t['description_not_available']})
                st.markdown(f"#### {details['subtitle']}")
                st.markdown(f"_{details['description']}_")
                st.image(os.path.join(results_dir, img_file), use_container_width=True)


    elif page == t['page_image_prediction']:
        st.header(t['prediction_header'])
        st.markdown(t['prediction_description'])

        # Selección dinámica de modelos
        selected_model_name = st.selectbox("Selecciona un modelo para la predicción:", list(models_dict.keys()))
        selected_model = models_dict[selected_model_name]

        col_image_upload, col_prediction_display = st.columns(2)

        with col_image_upload:
            uploaded_file = st.file_uploader(t['choose_image'], type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption=t['uploaded_image'], use_container_width=True)

        with col_prediction_display:
            if uploaded_file is not None and st.button(t['perform_prediction']):
                st.subheader(t['prediction_results_subheader'])
                
                predicted_class_english, confidence, probabilities = predict(image, selected_model)
                predicted_class_translated = t['class_names_map'][predicted_class_english]
                
                st.markdown(f"**{t['detected_disease']}** **{predicted_class_translated}**")
                st.markdown(f"<p style='color:green; font-size:20px;'>**{t['confidence']}: {confidence:.2f}%**</p>", unsafe_allow_html=True)

                # Gráfico de barras para las probabilidades
                fig, ax = plt.subplots(figsize=(8, 5))
                translated_class_names_for_plot = [t['class_names_map'][name] for name in CLASS_NAMES_ENGLISH]
                ax.bar(translated_class_names_for_plot, probabilities, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'brown', 'pink'])
                ax.set_ylabel(t['probability'])
                ax.set_title(t['class_probabilities'])
                ax.set_xticklabels(translated_class_names_for_plot, rotation=45, ha='right', fontsize=10)
                ax.tick_params(axis='y', labelsize=10)
                st.pyplot(fig)
                plt.close(fig)

                # Guardar la predicción en el historial
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model': selected_model_name,
                    'predicted_class': predicted_class_english,
                    'confidence': f"{confidence:.2f}%",
                    'probabilities': probabilities
                })

                # Opción para descargar la predicción actual
                prediction_data = {
                    'model_used': selected_model_name,
                    'predicted_class': predicted_class_english,
                    'confidence': f"{confidence:.2f}%",
                    'probabilities': {CLASS_NAMES_ENGLISH[i]: prob for i, prob in enumerate(probabilities)}
                }
                st.download_button(
                    label=t['download_current_prediction'],
                    data=json.dumps(prediction_data, indent=4),
                    file_name=f"prediction_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

                # Mostrar recomendaciones específicas para la enfermedad detectada
                st.markdown("---")
                st.subheader(t['specific_recommendations_title'])
                st.markdown(f"**{t['specific_recommendations']} {predicted_class_translated}:**")
                
                recommendations = {
                    'Bacteria': t['bacteria_reco'],
                    'Fungi': t['fungi_reco'],
                    'Healthy': t['healthy_reco'],
                    'Nematode': t['nematode_reco'],
                    'Pest': t['pest_reco'],
                    'Phytophthora': t['phytophthora_reco'],
                    'Virus': t['virus_reco']
                }
                
                reco_text = recommendations.get(predicted_class_english, t['no_specific_reco'])
                st.info(reco_text)

                # Se eliminó el bloque de código que mostraba los resultados de evaluación completos aquí.

    elif page == t['page_generate_report']:
        st.header(t['generate_report_header'])
        st.markdown(t['generate_report_description'])

        if st.button(t['generate_report_button']):
            st.info(t['generating_report'])
            
            # Regenerar los gráficos en el idioma actual antes de generar el PDF
            try:
                # Ejecutar plot_results.py como un subproceso
                # Asegúrate de que el entorno de Python sea el mismo que el de Streamlit
                import subprocess
                python_executable = sys.executable # Obtiene la ruta del ejecutable de Python actual
                command = [python_executable, "scripts/plot_results.py", "--lang", st.session_state.current_lang_code] # Usar st.session_state.current_lang_code
                
                # Capturar la salida para depuración si es necesario
                process = subprocess.run(command, capture_output=True, text=True, check=True)
                print("Salida de plot_results.py (stdout):", process.stdout)
                if process.stderr:
                    print("Salida de plot_results.py (stderr):", process.stderr)

            except subprocess.CalledProcessError as e:
                st.error(f"Error al generar los gráficos en el idioma seleccionado: {e.stderr}")
                print(f"Error al generar los gráficos: {e.stderr}")
                return
            except FileNotFoundError:
                st.error("Error: No se encontró el script 'scripts/plot_results.py'. Asegúrate de que el archivo existe.")
                return
            
            # Call the PDF generation function here
            generate_pdf_report(t, st.session_state.current_lang_code) # Pasar traducciones y current_lang_code
            st.success(t['report_generated_successfully'])
            st.download_button(
                label=t['download_report'],
                data=open("results/reporte_enfermedades_patata.pdf", "rb").read(),
                file_name="reporte_enfermedades_patata.pdf",
                mime="application/pdf"
            )
    
    elif page == t['page_disease_info']:
        st.header(t['disease_info_header'])
        st.markdown(t['disease_info_intro'])

        disease_info_data = {
            t['bacteria_name']: t['bacteria_desc'],
            t['fungi_name']: t['fungi_desc'],
            t['healthy_name']: t['healthy_desc'],
            t['nematode_name']: t['nematode_desc'],
            t['pest_name']: t['pest_desc'],
            t['phytophthora_name']: t['phytophthora_desc'],
            t['virus_name']: t['virus_desc']
        }

        col1, col2 = st.columns(2)
        disease_keys = list(disease_info_data.keys())
        
        for i, disease in enumerate(disease_keys):
            with (col1 if i % 2 == 0 else col2):
                with st.expander(disease):
                    st.markdown(disease_info_data[disease])
        
        st.markdown("---")
        st.header(t['recommendations_header'])
        st.markdown(t['recommendations_intro'])

        st.subheader(t['general_recommendations'])
        st.markdown(f"- {t['general_recommendation_1']}")
        st.markdown(f"- {t['general_recommendation_2']}")
        st.markdown(f"- {t['general_recommendation_3']}")
        st.markdown(f"- {t['general_recommendation_4']}")
        st.markdown(f"- {t['general_recommendation_5']}")

    elif page == t['prediction_history_header']:
        st.header(t['prediction_history_header'])
        if st.session_state.prediction_history:
            if st.button(t['clear_history']):
                st.session_state.prediction_history = []
                st.rerun() # Rerun to clear display
            
            for i, pred in enumerate(reversed(st.session_state.prediction_history)):
                st.subheader(f"{t['prediction_history_header'].split(' ')[0]} {len(st.session_state.prediction_history) - i} ({pred['timestamp']})")
                st.write(f"**{t['model_used_pdf']}** {pred['model']}")
                st.write(f"**{t['detected_disease_pdf']}** {t['class_names_map'][pred['predicted_class']]}")
                st.write(f"**{t['confidence_pdf']}** {pred['confidence']}")
                
                # Display probabilities as a table
                prob_data = [[t['class_names_map'][CLASS_NAMES_ENGLISH[j]], f"{prob:.2f}%"] for j, prob in enumerate(pred['probabilities'])]
                st.table(prob_data)
                st.markdown("---")
        else:
            st.info(t['no_predictions_yet'])


# Define a custom page template for headers/footers
def footer(canvas, doc, t): # Pass translations to footer
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 0.75 * inch, f"{t['page_text']} {doc.page}") # Use translated text
    canvas.restoreState()

def generate_pdf_report(t, lang_code): # Pass translations and lang_code to generate_pdf_report
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
    elements.append(Paragraph(t['report_title_pdf'], styles['TitleStyle']))
    elements.append(Paragraph(t['app_title_suffix_pdf'], styles['TitleStyle'])) # Use translated suffix
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(t['report_subtitle_pdf'], styles['SubtitleStyle']))
    elements.append(Spacer(1, 1.0 * inch))
    elements.append(Paragraph(f"{t['generation_date_pdf']} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['DateStyle']))
    elements.append(PageBreak())

    results_dir = 'results'
    model_names = ['ResNet18', 'ResNet50', 'DenseNet121']

    # Define image_details here so it's always available
    image_details = {
        'accuracy_per_model.png': {
            'subtitle': t['accuracy_per_model_subtitle'],
            'description': t['accuracy_per_model_desc']
        },
        'performance_comparison_accuracy.png': {
            'subtitle': t['performance_comparison_accuracy_subtitle'],
            'description': t['performance_comparison_accuracy_desc']
        },
        'training_time_comparison.png': {
            'subtitle': t['training_time_comparison_subtitle'],
            'description': t['training_time_comparison_desc']
        },
        'error_histogram.png': {
            'subtitle': t['error_histogram_subtitle'],
            'description': t['error_histogram_desc']
        },
        'prediction_correlation_matrix.png': {
            'subtitle': t['prediction_correlation_matrix_subtitle'],
            'description': t['prediction_correlation_matrix_desc']
        },
        'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png': {
            'subtitle': t['prediction_correlation_matrix_resnet18_densenet121_subtitle'],
            'description': t['prediction_correlation_matrix_resnet18_densenet121_desc']
        },
        'prediction_correlation_matrix_ResNet18_vs_ResNet50.png': {
            'subtitle': t['prediction_correlation_matrix_resnet18_resnet50_subtitle'],
            'description': t['prediction_correlation_matrix_resnet18_resnet50_desc']
        },
        'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png': {
            'subtitle': t['prediction_correlation_matrix_resnet50_densenet121_subtitle'],
            'description': t['prediction_correlation_matrix_resnet50_densenet121_desc']
        },
        'training_validation_plot_DenseNet121.png': {
            'subtitle': f"{t['results_for']} DenseNet121",
            'description': f"{t['training_validation_plot_desc']} DenseNet121"
        },
        'training_validation_plot_ResNet18.png': {
            'subtitle': f"{t['results_for']} ResNet18",
            'description': f"{t['training_validation_plot_desc']} ResNet18"
        },
        'training_validation_plot_ResNet50.png': {
            'subtitle': f"{t['results_for']} ResNet50",
            'description': f"{t['training_validation_plot_desc']} ResNet50"
        }
    }

    # 1. Historial de Predicciones
    elements.append(Paragraph(t['prediction_history_header'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))
    if st.session_state.prediction_history:
        for i, pred in enumerate(st.session_state.prediction_history):
            elements.append(Paragraph(f"<b>Predicción {i+1} ({pred['timestamp']}):</b>", styles['h4']))
            elements.append(Paragraph(f"<b>{t['model_used_pdf']}</b> {pred['model']}", styles['Normal']))
            elements.append(Paragraph(f"<b>{t['detected_disease_pdf']}</b> {t['class_names_map'][pred['predicted_class']]}", styles['Normal']))
            elements.append(Paragraph(f"<b>{t['confidence_pdf']}</b> {pred['confidence']}", styles['Normal']))
            
            # Tabla de probabilidades
            prob_data = [[t['class_names_map'][CLASS_NAMES_ENGLISH[j]], f"{prob:.2f}%"] for j, prob in enumerate(pred['probabilities'])]
            prob_table_data = [[t['class_pdf'], t['probability_pdf']]] + prob_data
            prob_table = Table(prob_table_data, colWidths=[2*inch, 2*inch])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(prob_table)

            # Recomendación para la predicción
            recommendations_data_pdf = {
                'Bacteria': t['bacteria_reco'],
                'Fungi': t['fungi_reco'],
                'Healthy': t['healthy_reco'],
                'Nematode': t['nematode_reco'],
                'Pest': t['pest_reco'],
                'Phytophthora': t['phytophthora_reco'],
                'Virus': t['virus_reco']
            }
            predicted_class_name_english = pred['predicted_class']
            specific_reco_text = recommendations_data_pdf.get(predicted_class_name_english, t['no_specific_reco'])
            
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph(f"<b>{t['specific_recommendation_for_pdf']} {t['class_names_map'][predicted_class_name_english]}:</b>", styles['h4']))
            elements.append(Paragraph(specific_reco_text, styles['Normal']))
            elements.append(PageBreak())
    else:
        elements.append(Paragraph(t['no_predictions_yet'], styles['Normal']))

    # 2. Recomendaciones Generales
    elements.append(Paragraph(t['recommendations_header'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(t['recommendations_intro'], styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(t['general_recommendations'], styles['h3']))
    elements.append(Paragraph(f"- {t['general_recommendation_1']}", styles['Normal']))
    elements.append(Paragraph(f"- {t['general_recommendation_2']}", styles['Normal']))
    elements.append(Paragraph(f"- {t['general_recommendation_3']}", styles['Normal']))
    elements.append(Paragraph(f"- {t['general_recommendation_4']}", styles['Normal']))
    elements.append(Paragraph(f"- {t['general_recommendation_5']}", styles['Normal']))
    elements.append(PageBreak())

    # 3. Características de los Modelos
    elements.append(Paragraph(t['model_features_section'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))
    model_names_list = ['ResNet18', 'ResNet50', 'DenseNet121']
    model_descriptions_pdf = {
        'ResNet18': t['resnet18_desc'],
        'ResNet50': t['resnet50_desc'],
        'DenseNet121': t['densenet121_desc']
    }
    for model_name in model_names_list:
        elements.append(Paragraph(f"<b>{model_name}</b>", styles['h3']))
        elements.append(Paragraph(model_descriptions_pdf[model_name], styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))
    elements.append(PageBreak())

    # 4. Tiempos de Entrenamiento
    elements.append(Paragraph(t['training_time_comparison_subtitle'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))
    img_file = 'training_time_comparison.png'
    details = image_details.get(img_file, {'subtitle': t['training_time_comparison_subtitle'], 'description': t['training_time_comparison_desc']})
    elements.append(Paragraph(details['subtitle'], styles['h3']))
    elements.append(Paragraph(details['description'], styles['Italic']))
    elements.append(RLImage(os.path.join(results_dir, img_file), width=6*inch, height=4*inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(PageBreak())

    # 5. Comparación de Modelos (Matrices de Confusión, Informes de Clasificación y MCC)
    elements.append(Paragraph(t['model_comparison_section'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))
    for model_name in model_names:
        elements.append(Paragraph(f"{t['results_for']} {model_name}:", styles['h3']))
        
        # Confusion Matrix Image
        cm_path = os.path.join(results_dir, f'confusion_matrix_{model_name.lower()}.png')
        if os.path.exists(cm_path):
            elements.append(RLImage(cm_path, width=4*inch, height=3.2*inch))
            elements.append(Paragraph(f"{t['confusion_matrix_caption']} {model_name}", styles['Italic']))
            elements.append(Spacer(1, 0.1 * inch))

        # Matthews Correlation Coefficient
        eval_json_path = os.path.join(results_dir, f'evaluation_results_potato_leaf_disease_model_{model_name.lower()}.json')
        if os.path.exists(eval_json_path):
            with open(eval_json_path, 'r') as f:
                eval_data = json.load(f)
                if 'matthews_corrcoef' in eval_data:
                    elements.append(Paragraph(f"{t['mcc_caption']} {model_name}:", styles['h4']))
                    elements.append(Paragraph(f"MCC: {eval_data['matthews_corrcoef']:.4f}", styles['Normal']))
                    elements.append(Spacer(1, 0.1 * inch))
        elements.append(PageBreak())

    # 6. Parte Estadística con Gráficos (McNemar y otros gráficos de rendimiento)
    elements.append(Paragraph(t['statistical_analysis_plots_section'], styles['h1']))
    elements.append(Spacer(1, 0.2 * inch))

    # McNemar's Test Results
    elements.append(Paragraph(t['mcnemar_test_section'], styles['h2']))
    elements.append(Spacer(1, 0.1 * inch))
    mcnemar_results_path = os.path.join(results_dir, 'mcnemar_test_results.json')
    if os.path.exists(mcnemar_results_path):
        with open(mcnemar_results_path, 'r') as f:
            mcnemar_results = json.load(f)
        
        data = [[t['model1'], t['model2'], t['chi_squared_statistic_table'], t['p_value_table']]]
        for result in mcnemar_results:
            model1_name = result['model1']
            model2_name = result['model2']

            if 'statistic' in result['results']:
                statistic = f"{result['results']['statistic']:.4f}"
                pvalue = f"{result['results']['pvalue']:.4f}"
            else:
                statistic = "N/A"
                pvalue = "N/A"
            data.append([model1_name, model2_name, statistic, pvalue])

        col_widths = [1.5 * inch, 1.5 * inch, 1.5 * inch, 2.0 * inch]

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ])
        
        table = Table(data, colWidths=col_widths)
        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(t['conclusions_mcnemar'], styles['h3']))
        for result in mcnemar_results:
            if 'statistic' in result['results']:
                conclusion_text = f"• {t['comparison']} {result['model1']} vs {result['model2']}: {result['results']['conclusion']}"
            else:
                conclusion_text = f"• {t['comparison']} {result['model1']} vs {result['model2']}: {t['error_performing_test']} {result['results']['error']}"
            elements.append(Paragraph(conclusion_text, styles['Normal']))
            elements.append(Spacer(1, 0.1 * inch))
    else:
        elements.append(Paragraph(t['no_mcnemar_results_pdf'], styles['Normal']))
    elements.append(PageBreak())

    # Other Performance Graphs
    elements.append(Paragraph(t['other_performance_graphs'], styles['h2']))
    
    # Filter out specific confusion matrices and training plots
    all_image_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
    specific_cm_files = [f'confusion_matrix_{name.lower()}.png' for name in model_names]
    plots_to_exclude_pdf = [
        'training_validation_plot_DenseNet121.png',
        'training_validation_plot_ResNet18.png',
        'training_validation_plot_ResNet50.png',
        'training_time_comparison.png', # Excluir de aquí porque ya se incluyó arriba
        'prediction_correlation_matrix.png', # Excluir esta imagen
        'prediction_correlation_matrix_ResNet18_vs_DenseNet121.png',
        'prediction_correlation_matrix_ResNet18_vs_ResNet50.png',
        'prediction_correlation_matrix_ResNet50_vs_DenseNet121.png'
    ]

    other_image_files = [f for f in all_image_files if f not in specific_cm_files and f not in plots_to_exclude_pdf]
    sorted_other_image_files = sorted(other_image_files)

    for img_file in sorted_other_image_files:
        details = image_details.get(img_file, {'subtitle': img_file.replace('_', ' ').replace('.png', ''), 'description': t['description_not_available']})
        elements.append(Paragraph(details['subtitle'], styles['h3']))
        elements.append(Paragraph(details['description'], styles['Italic']))
        elements.append(RLImage(os.path.join(results_dir, img_file), width=6*inch, height=4*inch))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(PageBreak())

    # Incluir las curvas ROC en el PDF
    for model_name in model_names:
        roc_path = os.path.join(results_dir, f'roc_curve_{model_name}.png')
        if os.path.exists(roc_path):
            elements.append(Paragraph(f"{t['roc_curve_title']} {model_name}", styles['h3']))
            elements.append(RLImage(roc_path, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(PageBreak())

    # Función para determinar el mejor modelo
    def get_best_model_recommendation(t):
        model_accuracies = {}
        model_training_times = {}
        
        for model_name in model_names:
            eval_path = os.path.join(results_dir, f'evaluation_results_potato_leaf_disease_model_{model_name.lower()}.json')
            history_path = os.path.join(results_dir, f'training_history_potato_leaf_disease_model_{model_name.lower()}.json')

            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                    accuracy = np.sum(np.array(eval_data['correct_predictions'])) / len(eval_data['correct_predictions'])
                    model_accuracies[model_name] = accuracy
            
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                    if 'training_time' in history_data:
                        model_training_times[model_name] = history_data['training_time']

        best_model = None
        best_accuracy = -1
        lowest_training_time = float('inf')

        # Encontrar el modelo con la mejor precisión
        for model, acc in model_accuracies.items():
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
            elif acc == best_accuracy:
                # Si las precisiones son iguales, preferir el que tenga menor tiempo de entrenamiento
                if model in model_training_times and best_model in model_training_times:
                    if model_training_times[model] < model_training_times[best_model]:
                        best_model = model
                elif best_model is None: # Si es el primer modelo con esta precisión
                    best_model = model

        recommendation_text = ""
        if best_model:
            recommendation_text += f"{t['best_model_intro']} **{best_model}** {t['best_model_middle']}. "
            recommendation_text += f"{t['accuracy_achieved']} **{model_accuracies.get(best_model, 0.0):.2f}**."
            if best_model in model_training_times:
                recommendation_text += f" {t['training_time_was']} **{model_training_times[best_model]:.2f} {t['seconds']}**, {t['making_it_efficient']}."
            
            # Mencionar la prueba de McNemar
            recommendation_text += f" {t['mcnemar_test_note']}"
        else:
            recommendation_text = t['no_best_model_found']
        
        return recommendation_text

    doc.build(elements, onFirstPage=lambda canvas, doc: footer(canvas, doc, t), onLaterPages=lambda canvas, doc: footer(canvas, doc, t))

if __name__ == '__main__':
    main()
