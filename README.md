# Detección de Enfermedades en Hojas de Patata 🥔🌿

Este repositorio contiene un sistema completo para la detección y clasificación de enfermedades en hojas de patata utilizando modelos de aprendizaje profundo. El proyecto abarca desde el entrenamiento de modelos hasta una interfaz web interactiva para la predicción en tiempo real y la generación de informes detallados.

## Características Principales ✨

*   **Entrenamiento de Modelos:** Capacita modelos de redes neuronales convolucionales (CNN) como ResNet18, ResNet50 y DenseNet121 en el dataset de enfermedades de la hoja de patata.
*   **Evaluación Exhaustiva:** Genera métricas de rendimiento detalladas, matrices de confusión, informes de clasificación y realiza pruebas estadísticas (e.g., Prueba de McNemar) para comparar el desempeño de los modelos.
*   **Interfaz Web Interactiva (Streamlit):** Una aplicación web fácil de usar para cargar imágenes de hojas de patata y obtener predicciones instantáneas de los modelos entrenados.
*   **Generación de Reportes PDF:** Crea informes PDF completos con todos los resultados estadísticos y gráficos de evaluación para una documentación y análisis sencillos.

## Configuración del Entorno 🛠️

Para poner en marcha el proyecto, sigue estos pasos:

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/Nex7G3n/potato-leaf-diseases.git
    cd potato-leaf-diseases
    ```

2.  **Instalar Dependencias:**
    Asegúrate de tener Python 3.8+ instalado. Luego, instala todas las librerías necesarias:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Autenticación de Kaggle (para descarga de dataset):**
    Si el dataset no está presente, el script de entrenamiento lo descargará automáticamente. Para ello, necesitas autenticar la API de Kaggle. Coloca tu archivo `kaggle.json` (que puedes obtener de tu perfil de Kaggle) en el directorio `~/.kaggle/` (en Windows, esto suele ser `C:\Users\TuUsuario\.kaggle\`).

## Estructura del Proyecto 📂

*   `app.py`: El script principal de la aplicación web Streamlit.
*   `data/`: Directorio donde se almacenará el dataset de imágenes.
*   `models/`: Contiene los modelos entrenados (`.pth`).
*   `results/`: Almacena todos los gráficos de evaluación, informes y resultados estadísticos.
*   `scripts/`: Scripts para el entrenamiento (`train.py`), evaluación (`evaluate.py`) y visualización de resultados (`plot_results.py`).
*   `requirements.txt`: Lista de dependencias de Python.

## Entrenamiento de Modelos 🚀

Para entrenar los modelos en el dataset de enfermedades de la hoja de patata, ejecuta el siguiente comando:

```bash
python scripts/train.py --data-dir data --epochs 10 --model-name resnet18 # o resnet50, densenet121
```
*   `--data-dir`: Ruta al directorio donde se guardará o ya se encuentra el dataset.
*   `--epochs`: Número de épocas para el entrenamiento.
*   `--model-name`: Especifica la arquitectura del modelo a entrenar (e.g., `resnet18`, `resnet50`, `densenet121`).

## Evaluación de Modelos 📊

Después de entrenar los modelos, puedes evaluarlos y generar métricas de rendimiento:

```bash
python scripts/evaluate.py --data-dir data --model-name resnet18 # o resnet50, densenet121
```
Este script generará informes de clasificación, matrices de confusión y archivos JSON con métricas detalladas en el directorio `results/`.

Para generar gráficos adicionales de comparación y la prueba de McNemar, ejecuta:

```bash
python scripts/plot_results.py
```
Este script creará varios archivos `.png` y un archivo `mcnemar_test_results.json` en el directorio `results/`.

## Interfaz Web Interactiva (Streamlit) 🌐

La aplicación web Streamlit permite interactuar con los modelos entrenados para realizar predicciones en tiempo real.

Para iniciar la aplicación, ejecuta:

```bash
streamlit run app.py
```

Una vez iniciada, la aplicación se abrirá en tu navegador web. La interfaz está dividida en varias secciones:

### 1. Inicio y Modelos

Esta sección proporciona una bienvenida, información general sobre el proyecto, detalles del dataset utilizado (incluyendo una distribución de clases) y descripciones de las arquitecturas de los modelos (ResNet18, ResNet50, DenseNet121).

![Screenshot of Home and Models Page](images/streamlit_home_models.png)
*Captura de pantalla de la página de Inicio y Modelos.*

### 2. Resultados de Evaluación

Aquí se muestran los resultados detallados de la evaluación de cada modelo, incluyendo matrices de confusión, informes de clasificación, el Coeficiente de Correlación de Matthews (MCC) y los resultados de la Prueba de McNemar. También se presentan gráficos comparativos de rendimiento.

![Screenshot of Evaluation Results Page](images/streamlit_evaluation_results.png)
*Captura de pantalla de la página de Resultados de Evaluación.*

### 3. Predicción de Imagen

En esta sección, puedes subir una imagen de una hoja de patata. La aplicación utilizará los modelos entrenados para predecir la enfermedad presente y mostrará la clase detectada junto con el nivel de confianza y un gráfico de probabilidades por clase.

![Screenshot of Image Prediction Page](images/streamlit_image_prediction.png)
*Captura de pantalla de la página de Predicción de Imagen.*

### 4. Generar Reporte PDF

Esta funcionalidad permite generar un informe PDF completo que consolida todos los resultados estadísticos y gráficos de evaluación del proyecto. El informe es útil para la documentación y el análisis fuera de la aplicación.

![Screenshot of PDF Report Generation Page](images/streamlit_pdf_report.png)
*Captura de pantalla de la página de Generación de Reporte PDF.*

## Contribuciones 🤝

Las contribuciones son bienvenidas. Si tienes alguna sugerencia o mejora, no dudes en abrir un 'issue' o enviar un 'pull request'.

## Licencia 📄

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
