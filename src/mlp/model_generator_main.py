"""Programa principal de generación de modelos (exportado de notebook)."""
import os
import sys
import pandas as pd

# Mantiene el ajuste de sys.path del script original.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from cashia_model.promok_model import *
from mlp.model_generator_functions import ModelGenerator

def main():

    configurations = [
        "NV_Agt",
        "RNV_Agt",
        "NV_CC",
        "RNV_CC",
        "NV_Agt_CS",
        "RNV_Agt_CS",
        "NV_CC_CS",
        "RNV_CC_CS"
        ]

    feature_engineering = "standardization"
    # feature_engineering = "normalization"

    local_notes = "Usando: "+feature_engineering+"Sin remplazar campos vacios por 'Unknown'"
    zip_digits = 4

    notes = "zip_digits = "+str(zip_digits)+"  "+local_notes

    # Fecha máxima de los datos para el aprendizaje y evaluación
    min_date =  pd.to_datetime('2023-01-01')
    max_date =  pd.to_datetime('2026-05-31')
    
    source_file_name = 'Data Set Cashia Entrenamiento.csv'
    configuration_file_name = "CashIA_ConfFile.xlsx"
    stats_file_name = "Statistics.xlsx"
    test_stats_file_name = "StatisticsTest.xlsx"
    test_size = 0.2



    for conf_column in configurations:

        print(f"\n{'=' * 60}")
        print(f"Training configuration: {conf_column}")
        print(f"{'=' * 60}")

        models_to_test = [#ModelToTest("Random_Forest_SMOTE", {'criterion':'gini', 'bootstrap':True, 'random_state':100}, model_conf=conf_column),
                ModelToTest("Random_Forest", {'criterion':'gini', 'bootstrap':True, 'random_state':100}, model_conf=conf_column),
                #ModelToTest("Regresion", {'random_state':0, 'solver':'newton-cg', 'max_iter':4000}, model_conf=conf_column),
                #ModelToTest("Naive Bayes", {}, model_conf=conf_column),
                #ModelToTest("Ada Boost", {'criterion':'entropy', 'max_depth':1, 'random_state':1, 
                #                          'n_estimators':500, 'learning_rate':0.1,'random_state':1}, model_conf=conf_column),
                #ModelToTest("SVC", {'kernel':'linear'}, model_conf=conf_column),
                #ModelToTest("KNN", {'n_neighbors':5}, model_conf=conf_column),
                #ModelToTest("KNN", {'n_neighbors':20}, model_conf=conf_column),
                #ModelToTest("KNN", {'n_neighbors':21}, model_conf=conf_column),
                #ModelToTest("KNN", {'n_neighbors':22}, model_conf=conf_column),
                #ModelToTest("KNN", {'n_neighbors':23}, model_conf=conf_column),
                ]

        generator = ModelGenerator(conf_column, 
                                   zip_digits, 
                                   source_file_name, 
                                   configuration_file_name, 
                                   stats_file_name,
                                   notes, min_date, 
                                   max_date, 
                                   test_size, 
                                   models_to_test, 
                                   feature_engineering)
        generator.run()


if __name__ == "__main__":
    main()
