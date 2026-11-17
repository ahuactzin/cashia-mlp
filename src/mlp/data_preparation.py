"""
Programa principal para la preparación de datos de CashIA.
"""

import mlp.promok_models_config as pkconfig
from mlp.data_preparation_functions import DataPreparation


def main():
    """Ejecuta el proceso de preparación de datos."""

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


    source_file_name = 'Data Set Cashia Entrenamiento.csv'
    configuration_file_name = "CashIA_ConfFile.xlsx"

    for conf_column in configurations:
        print(f"\n{'=' * 60}")
        print(f"Data preparation: {conf_column}")
        print(f"{'=' * 60}")

        preparation = DataPreparation(conf_column, configuration_file_name, source_file_name)
        preparation.run()


if __name__ == "__main__":
    main()