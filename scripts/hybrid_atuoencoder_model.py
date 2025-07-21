import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class HybridAutoencoderModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridAutoencoderModel, self).__init__()
        # Cargar DenseNet121 pre-entrenado
        self.base_model = models.densenet121(pretrained=True)
        # Eliminar la capa clasificadora original
        self.base_model.classifier = nn.Identity()

        # La salida de DenseNet121.features es un tensor con 1024 canales.
        # Para las capas de autoencoder, necesitamos ajustar las dimensiones.
        # Asumimos que la salida de features es (batch_size, 1024, H, W)
        # donde H y W son 7x7 para una entrada de 224x224.

        # Capas de decodificación (autoencoder)
        # Conv2DTranspose(512, (3,3))
        # La salida de DenseNet121.features es (batch_size, 1024, 7, 7)
        self.decoder_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # UpSampling2D((2,2))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Conv2DTranspose(256, (3,3))
        self.decoder_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        # UpSampling2D((2,2))
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Capas de clasificación
        # La concatenación será entre la salida codificada (1024 canales) y la decodificada (256 canales)
        # Esto es un poco tricky porque las dimensiones espaciales pueden no coincidir.
        # Necesitamos que `encoded` y `decoded` tengan las mismas dimensiones espaciales para concatenar.
        # Si `encoded` es (B, 1024, 7, 7) y `decoded` es (B, 256, 28, 28) después de dos upsamples.
        # Necesitamos hacer un upsample de `encoded` o un downsample de `decoded`.
        # Optaremos por hacer un upsample de `encoded` para que coincida con `decoded` antes de concatenar.
        
        # La salida final del autoencoder (decoded) será de 28x28 si la entrada es 224x224 y la salida de densenet es 7x7
        # 7x7 -> upsample (14x14) -> upsample (28x28)
        
        # Para la concatenación, necesitamos que `encoded` tenga las mismas dimensiones espaciales que `decoded`.
        # `encoded` es (B, 1024, 7, 7). `decoded` es (B, 256, 28, 28).
        # Upsample `encoded` para que sea (B, 1024, 28, 28)
        self.upsample_encoded = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False) # 7*4 = 28

        self.fc1 = nn.Linear(1024 + 256, 1024) # 1024 (encoded) + 256 (decoded) canales
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.base_model.features(x) # Salida: (batch_size, 1024, H, W)

        # Codificado (encoded)
        encoded = features

        # Decodificado (decoded)
        decoded = self.decoder_conv1(encoded)
        decoded = self.relu(decoded)
        decoded = self.upsample1(decoded)
        decoded = self.decoder_conv2(decoded)
        decoded = self.relu(decoded)
        decoded = self.upsample2(decoded)

        # Asegurarse de que las dimensiones espaciales coincidan para la concatenación
        # Si decoded es (B, C_decoded, H_decoded, W_decoded)
        # y encoded es (B, C_encoded, H_encoded, W_encoded)
        # Necesitamos que H_decoded == H_encoded y W_decoded == W_encoded
        # La salida de DenseNet121.features para 224x224 es 7x7.
        # Después de dos upsamples (2x2 cada uno), decoded será 7*2*2 = 28x28.
        # Por lo tanto, upsampleamos encoded a 28x28.
        encoded_upsampled = self.upsample_encoded(encoded)
        
        # Concatenar
        merged = torch.cat((encoded_upsampled, decoded), dim=1) # Concatenar a lo largo de la dimensión de canales

        # Clasificación
        x = F.adaptive_avg_pool2d(merged, (1, 1)) # GlobalAveragePooling2D
        x = torch.flatten(x, 1) # Aplanar para la capa lineal
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
