import os
import zipfile
import json
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATASET_SLUG = "warcoder/potato-leaf-disease-dataset"
DATASET_DIRNAME = "Potato Leaf Disease Dataset in Uncontrolled Environment"

def authenticate_kaggle():
    """Carga kaggle.json desde el proyecto y configura las variables de entorno."""
    kaggle_path = Path("kaggle.json")
    if not kaggle_path.exists():
        raise FileNotFoundError("No se encontró kaggle.json en el directorio actual.")
    
    with open(kaggle_path, "r") as f:
        kaggle_token = json.load(f)
    
    os.environ["KAGGLE_USERNAME"] = kaggle_token["username"]
    os.environ["KAGGLE_KEY"] = kaggle_token["key"]


def download_dataset(root: Path):
    """Download dataset using Kaggle API if it is not present."""
    dataset_path = root / DATASET_DIRNAME
    if dataset_path.exists():
        print(f"Dataset already exists in {dataset_path}")
        return dataset_path

    authenticate_kaggle()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    root.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Kaggle…")
    api.dataset_download_files(DATASET_SLUG, path=str(root), unzip=True)
    print("Download complete")

    if not dataset_path.exists():
        zips = list(root.glob('*.zip'))
        if not zips:
            raise RuntimeError("Dataset was not downloaded correctly")
        with zipfile.ZipFile(zips[0], 'r') as zf:
            zf.extractall(root)
    return dataset_path


def create_dataloaders(data_dir: Path, batch_size: int = 32, val_split: float = 0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    
    # Para evaluación, usaremos todo el dataset como conjunto de prueba/validación
    # o una división si se especifica un val_split.
    # Aquí, para la evaluación, es mejor usar un conjunto de prueba explícito si existe,
    # o el conjunto de validación si no.
    # Asumiremos que el val_split se refiere a la división del dataset total para obtener un conjunto de validación.
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Si solo estamos evaluando, podemos cargar todo el dataset como "test_ds"
    # o usar el val_ds si el train.py ya lo dividió.
    # Para este propósito, usaremos el val_ds como nuestro conjunto de evaluación.
    _, val_ds = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return val_loader, dataset.classes


def evaluate_model(model, val_loader, device, class_names, model_arch_name):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    all_correct_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_correct_predictions.extend((preds == labels).cpu().numpy())

    # Calcular y guardar la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nMatriz de Confusión para {model_arch_name}:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Verdadera")
    plt.title(f"Matriz de Confusión para {model_arch_name}")
    cm_filename = f"confusion_matrix_{model_arch_name.lower()}.png"
    plt.savefig(Path("results") / cm_filename)
    plt.close() # Cierra la figura para liberar memoria
    print(f"\nMatriz de Confusión guardada como results/{cm_filename}")

    # Calcular y guardar el reporte de clasificación
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\nReporte de Clasificación para {model_arch_name}:")
    print(report)

    report_filename = f"classification_report_{model_arch_name.lower()}.txt"
    with open(Path("results") / report_filename, "w") as f:
        f.write(report)
    print(f"\nReporte de Clasificación guardado como results/{report_filename}")

    # Calcular precisión general
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nPrecisión General (Accuracy) para {model_arch_name}: {accuracy:.4f}")

    # Calcular y guardar el Coeficiente de Correlación de Matthews (MCC)
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"\nCoeficiente de Correlación de Matthews (MCC) para {model_arch_name}: {mcc:.4f}")
    
    return all_labels, all_preds, all_correct_predictions, mcc


def main(args):
    data_root = Path(args.data_dir)
    dataset_path = download_dataset(data_root)

    val_loader, class_names = create_dataloaders(dataset_path, batch_size=args.batch_size)

    num_classes = len(class_names)
    
    model_arch_name = args.model_arch # Obtener el nombre de la arquitectura del modelo

    # Seleccionar la arquitectura del modelo
    if model_arch_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_arch_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_arch_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Arquitectura de modelo no soportada: {model_arch_name}")

    # Cargar el modelo entrenado
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modelo cargado exitosamente desde {model_path}")

    all_labels, all_preds, all_correct_predictions, mcc = evaluate_model(model, val_loader, device, class_names, model_arch_name)
    
    # Convertir a arrays de numpy para facilitar el procesamiento y luego a listas para JSON
    all_labels = np.array(all_labels).tolist()
    all_preds = np.array(all_preds).tolist()
    all_correct_predictions = np.array(all_correct_predictions).tolist()

    # Guardar las predicciones y etiquetas para análisis posterior (correlación, etc.)
    evaluation_results = {
        'labels': all_labels,
        'predictions': all_preds,
        'correct_predictions': all_correct_predictions,
        'class_names': class_names,
        'matthews_corrcoef': mcc
    }
    results_filename = f"evaluation_results_{Path(args.model_path).stem}.json"
    with open(Path("results") / results_filename, 'w') as f:
        json.dump(evaluation_results, f)
    print(f"Resultados de evaluación guardados en results/{results_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Potato Leaf Disease Model")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store dataset')
    parser.add_argument('--model-path', type=str, default='models/potato_leaf_disease_model.pth', help='Path to the trained model .pth file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--model-arch', type=str, default='resnet18', help='Model architecture used for training (e.g., resnet18, resnet50)')
    args = parser.parse_args()
    main(args)
