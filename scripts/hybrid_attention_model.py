import torch
import torch.nn as nn
from torchvision import models

class HybridAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridAttentionModel, self).__init__()
        # Cargar EfficientNetB3 pre-entrenado
        self.base_model = models.efficientnet_b3(pretrained=True)
        # Eliminar la capa clasificadora original
        self.base_model.classifier = nn.Identity()

        # Capas de atención
        # La salida de EfficientNetB3 es (batch_size, channels, height, width)
        # Para EfficientNetB3, la salida antes del clasificador tiene 1536 canales.
        self.attention_conv1 = nn.Conv2d(1536, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention_conv2 = nn.Conv2d(64, 1536, kernel_size=1) # Canales de salida del base_model
        self.sigmoid = nn.Sigmoid()

        # Capas de clasificación
        self.fc1 = nn.Linear(1536, 1024) # 1536 es el número de canales de salida del base_model
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        base_output = self.base_model.features(x) # Obtener las características del EfficientNet

        # Lógica de atención
        attention = self.attention_conv1(base_output)
        attention = self.relu(attention)
        attention = self.global_avg_pool(attention)
        # Reshape no es necesario si global_avg_pool ya da (batch, channels, 1, 1)
        attention = self.attention_conv2(attention)
        attention = self.sigmoid(attention)
        
        # Multiplicación elemento a elemento
        attended_output = base_output * attention

        # Clasificación
        x = self.global_avg_pool(attended_output)
        x = torch.flatten(x, 1) # Aplanar para la capa lineal
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x # No aplicar softmax aquí, CrossEntropyLoss lo hace internamente
