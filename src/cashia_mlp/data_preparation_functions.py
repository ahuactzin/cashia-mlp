"""Preparación de datos CashIA, refactorizada desde data_preparation.py.

Cada instancia procesa una columna de configuración. Se preserva el orden de
las transformaciones del notebook original.
"""
import os
from pathlib import Path
from datetime import datetime
import sys


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk


import cashia_core.data_channel.promok_datapipe_func as pkdp
import cashia_core.common_tools.pm_mxgeographic as pm_mxgeo

from cashia_core.common_tools.configuration.resource_keys import get_resource_path
from cashia_core.common_tools.storage import get_storage

from cashia_mlp.dataframe_tracker import DataFrameEvolutionTracker
from cashia_mlp.config import get_mlp_resource_key


BASE_DIR = Path(__file__).resolve().parent


class DataPreparation:
    """Pipeline de preparación de datos para una configuración específica."""

    def __init__(self, conf_column, configuration_file_name, source_file_name):
        self.conf_column = conf_column
        self.configuration_file_name = configuration_file_name
        self.source_file_name = source_file_name

    def run(self):
        """Ejecuta todas las etapas en el orden original y devuelve los datos."""
        print("numpy:", np.__version__)
        print("pandas:", pd.__version__)
        print("seborn:", sns.__version__)
        print("sklearn:", sk.__version__)
        self.load_configuration()
        self.load_source_data()
        self.filter_category()
        self.clean_data()
        self.aggregate_features()
        self.save_results()
        self.finish()
        return self.data

    def load_configuration(self):
        print('Configuration file: ', self.configuration_file_name)
        print('Configuration : ', self.conf_column)

        self.all_features = get_storage().read_excel(f"{get_mlp_resource_key('conf_dir')}{self.configuration_file_name}", index_col=0)

        self.features = self.all_features[self.all_features[self.conf_column] == 'Sí']
        self.features_to_use = set(self.features['Característica'].values)
        self.agregated_feautures = set(self.features[self.features['Sección'] == 'Data Agregation']['Característica'].values)

    def load_source_data(self):
        print('Source file: ', self.source_file_name)

        self.data = get_storage().read_csv(f"{get_mlp_resource_key('inputs_dir')}{self.source_file_name}", index_col=0)
        self.data_tracker = DataFrameEvolutionTracker(self.data)

        if self.conf_column == 'NV_Agt' or self.conf_column == 'RNV_Agt':
            print('Agent score')
            self.data = self.data[self.data['Score Agt'] != 0]
        else:
            print('Using CC score')
        self.data_tracker.register_step(f'First filter for {self.conf_column}', self.data)



    def filter_category(self):
        if self.conf_column == 'NV_Agt':
            self.data = self.data[self.data['Categoria'] == 'NV']
        elif self.conf_column == 'RNV_Agt':
            self.data = self.data[self.data['Categoria'] == 'RNV']
        elif self.conf_column == 'NV_CC':
            self.data = self.data[self.data['Categoria'] == 'NV']
        elif self.conf_column == 'RNV_CC':
            self.data = self.data[self.data['Categoria'] == 'RNV']
        elif self.conf_column == 'NV_Agt_CS':
            self.data = self.data[self.data['Categoria'] == 'NV']
        elif self.conf_column == 'RNV_Agt_CS':
            self.data = self.data[self.data['Categoria'] == 'RNV']
        elif self.conf_column == 'NV_CC_CS':
            self.data = self.data[self.data['Categoria'] == 'NV']
        elif self.conf_column == 'RNV_CC_CS':
            self.data = self.data[self.data['Categoria'] == 'RNV']
        else:
            print('Invalid option')
            exit()
        self.data_tracker.register_step(f'Second filter for {self.conf_column}', self.data)


    def clean_data(self):
        self.data_features = set(self.data.columns)
        self.features_to_drop = list(self.data_features - self.features_to_use)
        self.features_to_drop.remove('Fecha Inicial')
        self.int_features_list = self.features[(self.features['Sub tipo'] == 'Entero') & (self.features['Sección'] != 'Data Agregation')]
        self.int_features_list = self.int_features_list['Característica'].unique()
        self.int_features_list = list(self.int_features_list)
        if 'CP' not in self.features_to_use and ('CP_Latitud' in self.features_to_use) | ('CP_Longitud' in self.features_to_use):
            self.int_features_list.append('CP')
        if 'CP Aval' not in self.features_to_use and ('CP_Aval_Latitud' in self.features_to_use) | ('CP_Aval_Longitud' in self.features_to_use):
            self.int_features_list.append('CP Aval')
        if 'Score CC' not in self.features_to_use and 'Odds' in self.features_to_use:
            self.int_features_list.append('Score CC')
        self.data.info()

        self.quantitative_features = set(self.features[(self.features['Tipo de dato'] == 'Cuantitativo') & (self.features['Tipo de variable'] == 'Predictor')]['Característica'].values)
        self.categorical_features = set(self.features[(self.features['Tipo de dato'] == 'Categórico') & (self.features['Tipo de variable'] == 'Predictor')]['Característica'].values)
        if 'Ingreso Conyuge' in self.features_to_use:
            pkdp.clean_ingreso_conyuge(self.data)
        self.data_tracker.register_step(f'Cleaning datos conyuge', self.data)

        pkdp.clean_egresos(self.data)
        self.data_tracker.register_step(f'Cleaning egresos', self.data)


        self.raw_data = self.features_to_use - self.agregated_feautures

        self.data = self.data.dropna(subset=list(self.features_to_use - self.agregated_feautures))
        self.data_tracker.register_step(f'Cleaning for columns with null', self.data)


        self.data[self.int_features_list] = self.data[self.int_features_list].apply(pd.to_numeric, errors='coerce', downcast='integer')

        self.data = self.data.dropna(subset=self.int_features_list)
        self.data[self.int_features_list] = self.data[self.int_features_list].astype(int)
        self.data_tracker.register_step(f'Cleaning int features with dropna', self.data)

        self.float_features_list = self.features[(self.features['Sub tipo'] == 'Real') & (self.features['Sección'] != 'Data Agregation')]['Característica'].unique()
        self.float_features_list = list(self.float_features_list)
        self.data[self.float_features_list] = self.data[self.float_features_list].apply(lambda col: pd.to_numeric(col, errors='coerce'))
        self.data[self.float_features_list] = self.data[self.float_features_list].astype(float)
        self.data = self.data.dropna(subset=self.float_features_list)
        print(self.data[self.float_features_list].info())
        self.data_tracker.register_step(f'Cleaning na floats', self.data)

        for feature in self.quantitative_features:
            if feature not in self.agregated_feautures:
                pkdp.categorical_print_values(feature, self.data)
                print('-----------------------')

        for feature in self.categorical_features:
            if feature not in self.agregated_feautures:
                pkdp.categorical_print_values(feature, self.data)
                print('-----------------------')
        self.numeric_features = self.features[(self.features['Numérico'] == 1) & (self.features['Sección'] != 'Data Agregation')]
        print(self.numeric_features)
        pkdp.print_sizes(self.data)
        for index, row in self.numeric_features.iterrows():
            print(f"Column {row['Característica']} max value found: {self.data[row['Característica']].max()}")
            pkdp.clean_numeric_data(self.data, row['Característica'], row['Min'], row['Max'])
            print('=================================================')
        pkdp.print_sizes(self.data)
        self.data_tracker.register_step(f'Cleaning numeric data min max', self.data)



    def aggregate_features(self):
        if 'Deficit' in self.quantitative_features:
            self.data['Deficit'] = self.data['Ingresos'] - self.data['Egresos']
        if 'IngresoPercapita' in self.quantitative_features:
            self.data['IngresoPercapita'] = self.data['Ingresos'] / (self.data['Dependientes'] + 1)

        self.cp_manager = pm_mxgeo.MX_CP_manager(get_resource_path("cp_coord_zone"))

        if 'TipoZona' in self.categorical_features:
            self.data['index'] = self.data.index
            print(self.data.head(5))
            self.data = self.data.merge(self.cp_manager.cp_base[['CP', 'TipoZona']], left_on='CP', right_on='CP', how='left').fillna('Fake')
            self.data = self.data.set_index(['index'])
        if 'Tiene aval' in self.categorical_features:
            self.data['Tiene aval'] = self.data.apply(lambda x: 0 if pd.isnull(x['CP Aval']) else 1, axis=1)
        if self.features_to_use.intersection(pkdp.latitud_longitud_features_set):
            if 'CP_Latitud' in self.features_to_use:
                print('Seting CP_Latitud')
                self.data = self.cp_manager.set_geographic_coordinate(self.data, 'CP', 'CP_Latitud', 'Latitud')
                print(self.data.columns)
                print('-----------------------')
            if 'CP_Longitud' in self.features_to_use:
                print('Seting CP_Longitud')
                self.data = self.cp_manager.set_geographic_coordinate(self.data, 'CP', 'CP_Longitud', 'Longitud')
                print(self.data.columns)
                print('-----------------------')
            if 'CP_Aval_Latitud' in self.features_to_use:
                self.data = self.cp_manager.set_geographic_coordinate(self.data, 'CP Aval', 'CP_Aval_Latitud', 'Latitud')
                print(self.data.columns)
                print('-----------------------')
            if 'CP_Aval_Longitud' in self.features_to_use:
                self.data = self.cp_manager.set_geographic_coordinate(self.data, 'CP Aval', 'CP_Aval_Longitud', 'Longitud')
                print(self.data.columns)
                print('-----------------------')
        self.data.info()
        if 'Odds' in self.quantitative_features:
            self.score_bins = [-1, 235, 406, 433, 451, 465, 477, 489, 499, 509, 518, 528, 538, 548, 559, 571, 584, 598, 614, 634, 664, 861]
            self.odds = [0, 0.9, 1.2, 1.3, 1.5, 1.6, 1.8, 1.9, 2.0, 2.1, 2.2, 2.6, 2.75, 2.9, 3.2, 3.5, 4.0, 4.6, 5.3, 6.1, 8.3]
            self.data['Odds'] = pd.cut(self.data['Score CC'], self.score_bins, labels=self.odds)
            self.data['Odds'] = self.data['Odds'].astype(float)
            print(self.data[['Score CC', 'Odds']])
        if 'Mes inicial' in self.quantitative_features:
            self.data['Fecha Inicial'] = pd.to_datetime(self.data['Fecha Inicial'], format='%d/%m/%Y')
            self.data['Mes inicial'] = self.data['Fecha Inicial'].dt.month
        if 'Estudios' in self.features_to_use:
            self.nivel_educativo = {0: 'Sin Estudios', 1: 'Primaria', 2: 'Secundaria', 3: 'Preparatoria', 4: 'Técnica', 5: 'Licenciatura', 6: 'Maestría', 7: 'Carrera Trunca'}
            self.data['Estudios'] = self.data['Estudios'].map(self.nivel_educativo)

        if 'Plazo' in self.features_to_use:
            self.plazo_pago = {9: '9 Sem', 13: '13 Sem', 16: '16 Sem', 17: '17 Sem', 26: '26 Sem', 27: '27 Sem'}
            self.data['Plazo'] = self.data['Plazo'].map(self.plazo_pago)


    def save_results(self):
        self.data = self.data.drop(labels=self.features_to_drop, axis=1)

        pkdp.print_sizes(self.data)
        self.data.info()

        source_fn, _ = os.path.splitext(self.source_file_name)
        processed_data_file_name = source_fn+self.conf_column+"Processed.csv"
        
        get_storage().write_csv(f"{get_mlp_resource_key('processed_data_dir')}{processed_data_file_name}", self.data)


    def finish(self):
        time_now = datetime.now()
        format_date_hour = '%Y-%m-%d %H:%M:%S'
        date_and_hour = time_now.strftime(format_date_hour)
        print('Ended at:', date_and_hour)


        self.data = self.data.reset_index(drop=True)

        print('Finalizado a las:', datetime.now().strftime('%H:%M:%S'))
        if sys.platform.startswith('win'):
            import winsound
            winsound.Beep(1000, 500)
            winsound.Beep(500, 500)
            winsound.Beep(1000, 500)
        else:
            print('\x07')
        print('Configuration : ', self.conf_column)
        pkdp.print_sizes(self.data)
