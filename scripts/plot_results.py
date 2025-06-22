import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_history(history, model_name, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title(f'Validation Accuracy for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path / f"training_validation_plot_{model_name}.png")
    plt.close()
    print(f"Gráfico de entrenamiento/validación para {model_name} guardado en {save_path / f'training_validation_plot_{model_name}.png'}")

def plot_performance_comparison(histories, model_names, save_path):
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        epochs = range(1, len(history['val_acc']) + 1)
        plt.plot(epochs, history['val_acc'], label=f'{model_names[i]} Validation Accuracy')
    plt.title('Model Performance Comparison (Validation Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / "performance_comparison_accuracy.png")
    plt.close()
    print(f"Gráfico de rendimiento comparativo guardado en {save_path / 'performance_comparison_accuracy.png'}")

def plot_error_histogram(correct_predictions_list, model_names, save_path):
    plt.figure(figsize=(10, 6))
    for i, correct_preds in enumerate(correct_predictions_list):
        errors = [1 - p for p in correct_preds] # 1 for incorrect, 0 for correct
        plt.hist(errors, bins=[0, 0.5, 1], rwidth=0.8, alpha=0.6, label=f'{model_names[i]} Errors', align='left')
    
    plt.xticks([0.25, 0.75], ['Correctas', 'Incorrectas'])
    plt.title('Histograma de Errores Individuales de Predicción')
    plt.xlabel('Tipo de Predicción')
    plt.ylabel('Número de Predicciones')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path / "error_histogram.png")
    plt.close()
    print(f"Histograma de errores individuales guardado en {save_path / 'error_histogram.png'}")

def plot_accuracy_per_model(accuracies, model_names, save_path):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Exactitud por Modelo')
    plt.ylim(0.0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
    plt.savefig(save_path / "accuracy_per_model.png")
    plt.close()
    print(f"Gráfico de exactitud por modelo guardado en {save_path / 'accuracy_per_model.png'}")

def plot_prediction_correlation_matrix(predictions_list, model_names, class_names, save_path):
    if len(predictions_list) < 2:
        print("Se necesitan al menos dos conjuntos de predicciones para calcular la correlación.")
        return

    # Convertir las listas de predicciones a arrays de numpy
    preds_model1 = np.array(predictions_list[0])
    preds_model2 = np.array(predictions_list[1])

    # Crear una matriz de co-ocurrencia de predicciones
    # Esto es una forma de ver qué tan a menudo ambos modelos predicen la misma clase
    # o diferentes clases para la misma imagen.
    # Para una verdadera "matriz de correlación" entre predicciones, podríamos usar
    # el coeficiente de Kappa de Cohen o simplemente la concordancia.
    # Aquí, vamos a crear una matriz de confusión cruzada entre los dos modelos.

    # Asegurarse de que las predicciones tienen la misma longitud
    min_len = min(len(preds_model1), len(preds_model2))
    preds_model1 = preds_model1[:min_len]
    preds_model2 = preds_model2[:min_len]

    # Calcular la "matriz de confusión" entre las predicciones de los dos modelos
    # Donde las filas son las predicciones del Modelo 1 y las columnas las del Modelo 2
    cross_confusion = confusion_matrix(preds_model1, preds_model2, labels=range(len(class_names)))

    plt.figure(figsize=(12, 10))
    sns.heatmap(cross_confusion, annot=True, fmt="d", cmap="viridis", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel(f"Predicciones de {model_names[1]}")
    plt.ylabel(f"Predicciones de {model_names[0]}")
    plt.title('Matriz de Correlación de Predicciones entre Modelos')
    plt.savefig(save_path / "prediction_correlation_matrix.png")
    plt.close()
    print(f"Matriz de correlación de predicciones guardada en {save_path / 'prediction_correlation_matrix.png'}")


def main():
    results_dir = Path("results")
    
    # Definir los modelos y sus archivos
    models_info = [
        {'name': 'ResNet18', 'history_file': 'training_history_potato_leaf_disease_model_resnet18.json', 'eval_file': 'evaluation_results_potato_leaf_disease_model_resnet18.json'},
        {'name': 'ResNet50', 'history_file': 'training_history_potato_leaf_disease_model_resnet50.json', 'eval_file': 'evaluation_results_potato_leaf_disease_model_resnet50.json'}
    ]

    all_histories = []
    all_eval_results = []
    all_accuracies = []
    all_predictions = []
    all_correct_predictions_for_hist = []
    
    for model_info in models_info:
        model_name = model_info['name']
        
        # Cargar historial de entrenamiento
        history_path = results_dir / model_info['history_file']
        if history_path.exists():
            history = load_json_data(history_path)
            all_histories.append(history)
            plot_training_history(history, model_name, results_dir)
        else:
            print(f"Advertencia: Historial de entrenamiento no encontrado para {model_name} en {history_path}")
            all_histories.append(None) # Añadir None para mantener el índice

        # Cargar resultados de evaluación
        eval_path = results_dir / model_info['eval_file']
        if eval_path.exists():
            eval_results = load_json_data(eval_path)
            all_eval_results.append(eval_results)
            
            # Calcular precisión general para el gráfico de exactitud
            accuracy = np.sum(np.array(eval_results['correct_predictions'])) / len(eval_results['correct_predictions'])
            all_accuracies.append(accuracy)
            
            # Recopilar predicciones para la matriz de correlación
            all_predictions.append(eval_results['predictions'])
            
            # Recopilar correct_predictions para el histograma de errores
            all_correct_predictions_for_hist.append(eval_results['correct_predictions'])

        else:
            print(f"Advertencia: Resultados de evaluación no encontrados para {model_name} en {eval_path}")
            all_eval_results.append(None) # Añadir None para mantener el índice
            all_accuracies.append(0.0) # Añadir 0.0 para mantener el índice
            all_predictions.append([])
            all_correct_predictions_for_hist.append([])


    # Generar gráficos comparativos si hay suficientes datos
    valid_histories = [h for h in all_histories if h is not None]
    valid_model_names_hist = [models_info[i]['name'] for i, h in enumerate(all_histories) if h is not None]
    if len(valid_histories) > 0:
        plot_performance_comparison(valid_histories, valid_model_names_hist, results_dir)

    valid_eval_results = [e for e in all_eval_results if e is not None]
    valid_model_names_eval = [models_info[i]['name'] for i, e in enumerate(all_eval_results) if e is not None]
    
    if len(valid_eval_results) > 0:
        plot_accuracy_per_model(all_accuracies, valid_model_names_eval, results_dir)
        plot_error_histogram(all_correct_predictions_for_hist, valid_model_names_eval, results_dir)
        
        if len(all_predictions) >= 2:
            # Asumiendo que el primer modelo es ResNet18 y el segundo es ResNet50
            # Necesitamos los nombres de las clases de uno de los modelos (deberían ser los mismos)
            class_names = valid_eval_results[0]['class_names']
            plot_prediction_correlation_matrix(all_predictions, valid_model_names_eval, class_names, results_dir)
        else:
            print("Advertencia: No hay suficientes resultados de evaluación para generar la matriz de correlación de predicciones.")


if __name__ == "__main__":
    main()
