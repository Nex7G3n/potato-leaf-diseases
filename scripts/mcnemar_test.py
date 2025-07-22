import json
from pathlib import Path
import itertools
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_tests():
    model_names = [
        "densenet121",
        "hybrid_attention",
        "hybrid_autoencoder",
        "resnet18",
        "resnet50"
    ]

    model_correct_predictions = {}
    results_dir = Path("results")

    # Cargar las predicciones correctas para cada modelo
    for model_name in model_names:
        file_path = results_dir / f"evaluation_results_{model_name.lower()}.json"
        if not file_path.exists():
            print(f"Advertencia: Archivo no encontrado para {model_name}: {file_path}")
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            model_correct_predictions[model_name] = np.array(data['correct_predictions'])
    
    if len(model_correct_predictions) < 2:
        print("No hay suficientes modelos para realizar comparaciones.")
        return

    # Realizar la prueba de McNemar para cada par de modelos
    print("Resultados de la Prueba de McNemar entre pares de modelos:\n")
    
    for model1_name, model2_name in itertools.combinations(model_correct_predictions.keys(), 2):
        preds1 = model_correct_predictions[model1_name]
        preds2 = model_correct_predictions[model2_name]

        # Asegurarse de que las longitudes de las predicciones coincidan
        if len(preds1) != len(preds2):
            print(f"Advertencia: Las longitudes de las predicciones no coinciden para {model1_name} y {model2_name}. Saltando.")
            continue

        # Construir la tabla de contingencia 2x2 para McNemar
        # n01: modelo1 incorrecto, modelo2 correcto
        # n10: modelo1 correcto, modelo2 incorrecto
        n01 = np.sum((preds1 == False) & (preds2 == True))
        n10 = np.sum((preds1 == True) & (preds2 == False))

        # La prueba de McNemar se basa en una tabla 2x2 de desacuerdos
        # [[n00, n01],
        #  [n10, n11]]
        # Donde n00 y n11 no son necesarios para la prueba en sí, solo n01 y n10.
        # Sin embargo, la función mcnemar espera una tabla completa.
        # Podemos calcular n00 y n11 para completar la tabla si es necesario,
        # pero para la prueba de McNemar, solo importan los desacuerdos.
        # La implementación de statsmodels.stats.contingency_tables.mcnemar
        # puede tomar directamente los valores n01 y n10 si se usa el parámetro 'exact=False'
        # o si se le pasa una tabla 2x2.

        # Para la prueba exacta, necesitamos la tabla completa.
        # n00 = np.sum((preds1 == False) & (preds2 == False))
        # n11 = np.sum((preds1 == True) & (preds2 == True))
        # table = [[n00, n01], [n10, n11]]
        
        # Usaremos la forma más simple que se enfoca en los desacuerdos
        # La prueba de McNemar es para comparar clasificadores en el mismo conjunto de datos.
        # Se enfoca en los casos donde un clasificador es correcto y el otro no.
        
        # Si n01 + n10 es 0, no hay desacuerdos, por lo que el chi-cuadrado es 0 y p-value es 1.
        if n01 + n10 == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            # La prueba de McNemar se puede realizar directamente con n01 y n10
            # La función mcnemar de statsmodels puede tomar una tabla 2x2
            # donde la diagonal principal son los acuerdos y la secundaria los desacuerdos.
            # Para la prueba de McNemar, la tabla es:
            # [[acuerdos_incorrectos, desacuerdos_M1_incorrecto_M2_correcto],
            #  [desacuerdos_M1_correcto_M2_incorrecto, acuerdos_correctos]]
            # Sin embargo, la prueba de McNemar se enfoca en los desacuerdos.
            # La forma más común de la prueba de McNemar es:
            # chi2 = (abs(n01 - n10) - 1)^2 / (n01 + n10) para corrección de continuidad
            # chi2 = (n01 - n10)^2 / (n01 + n10) sin corrección de continuidad
            # La función mcnemar de statsmodels maneja esto.
            
            # Para usar mcnemar de statsmodels, necesitamos una tabla de contingencia
            # donde las filas son las predicciones del Modelo 1 (incorrecto/correcto)
            # y las columnas son las predicciones del Modelo 2 (incorrecto/correcto).
            # La tabla debe ser:
            # [[M1_inc_M2_inc, M1_inc_M2_corr],
            #  [M1_corr_M2_inc, M1_corr_M2_corr]]
            
            # Recalculamos para la tabla completa para mcnemar
            m1_inc_m2_inc = np.sum((preds1 == False) & (preds2 == False))
            m1_inc_m2_corr = np.sum((preds1 == False) & (preds2 == True))
            m1_corr_m2_inc = np.sum((preds1 == True) & (preds2 == False))
            m1_corr_m2_corr = np.sum((preds1 == True) & (preds2 == True))
            
            table = [[m1_inc_m2_inc, m1_inc_m2_corr],
                     [m1_corr_m2_inc, m1_corr_m2_corr]]

            result = mcnemar(table, exact=False) # exact=False para chi-cuadrado aproximado
            chi2_stat = result.statistic
            p_value = result.pvalue

        print(f"Comparación: {model1_name} vs {model2_name}")
        print(f"  Modelo 1 incorrecto, Modelo 2 correcto (n01): {n01}")
        print(f"  Modelo 1 correcto, Modelo 2 incorrecto (n10): {n10}")
        print(f"  Estadístico Chi-cuadrado: {chi2_stat:.4f}")
        print(f"  Valor P: {p_value:.4f}\n")

if __name__ == "__main__":
    run_mcnemar_tests()
