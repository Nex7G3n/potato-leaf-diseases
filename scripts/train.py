import os
import zipfile
import json
from pathlib import Path
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

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

    # 🔄 Paso 1: Cargar las credenciales manualmente
    authenticate_kaggle()

    # 🔄 Paso 2: Importar KaggleApi luego de haber cargado las variables de entorno
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    root.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Kaggle…")
    api.dataset_download_files(DATASET_SLUG, path=str(root), unzip=True)
    print("Download complete")

    if not dataset_path.exists():
        # fallback: check for zip and extract manually
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
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(dataset.classes)




def main(args):
    data_root = Path(args.data_dir)
    dataset_path = download_dataset(data_root)

    train_loader, val_loader, num_classes = create_dataloaders(dataset_path, batch_size=args.batch_size)

    # Seleccionar la arquitectura del modelo
    if args.model_arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model_arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Arquitectura de modelo no soportada: {args.model_arch}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) # Mover el modelo al dispositivo antes de inicializar el optimizador

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Almacenar historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

    torch.save(model.state_dict(), Path("models") / args.output_model_name)
    
    # Guardar el historial de entrenamiento
    history_path = Path("results") / f"training_history_{args.output_model_name.replace('.pth', '.json')}"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Historial de entrenamiento guardado en {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on Potato Leaf Disease Dataset")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to store dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--model-arch', type=str, default='resnet18', help='Model architecture to use (e.g., resnet18, resnet50)')
    parser.add_argument('--output-model-name', type=str, default='potato_leaf_disease_model.pth', help='Name of the output model .pth file')
    args = parser.parse_args()
    main(args)
