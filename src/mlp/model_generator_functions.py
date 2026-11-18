"""Funciones extraídas del notebook model_generator.

Se conserva el orden de las operaciones y la lógica de entrenamiento.
Las rutas relativas siguen dependiendo del directorio de trabajo original.
"""
import sys
import sqlite3
import json
from datetime import datetime
import matplotlib
import sklearn
matplotlib.use("Agg")  # Backend sin ventanas; antes de pyplot / mlp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer

import mlp.promok_functions as pk
from mlp.dataframe_tracker import DataFrameEvolutionTracker

import cashia_core.data_channel.promok_datapipe_func as pkdp
import cashia_core.data_channel.pm_datapipeline as pmdp
from cashia_core.common_tools.storage import get_storage

from mlp.model_generator_config import get_mlp_resource_key

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

source_file_name = 'Data Set Cashia Entrenamiento.csv'

class ModelGenerator:

    def __init__(self, conf_column, zip_digits, source_file_name, conf_file_name, stats_file_name,
                 notes, min_date, max_date, test_size, models_to_test,
                 feature_engineering):
        self.conf_column = conf_column
        self.zip_digits = zip_digits
        self.configuration_file_name = conf_file_name
        self.stats_file_name = stats_file_name
        self.notes = notes
        self.max_date = max_date
        self.min_date = min_date
        self.test_size = test_size
        self.models_to_test = models_to_test
        self.feature_engineering = feature_engineering
        self.source_file_name = source_file_name




    def load_configuration(self, all_stats):
        print('Configuration file: ', self.configuration_file_name)

        all_features = get_storage().read_excel(
            f"{get_mlp_resource_key('conf_dir')}{self.configuration_file_name}",
            index_col=0
)
        features = all_features[all_features[self.conf_column] == 'Sí']

        features_to_use = set(features['Característica'].values)

        categorical_features = set(features[(features['Tipo de dato'] == 'Categórico') & (features['Tipo de variable'] == 'Predictor')]['Característica'].values)

        # ### 1.2 Datos cuantitativos

        quantitative_features = set(features[(features['Tipo de dato'] == 'Cuantitativo') & (features['Tipo de variable'] == 'Predictor')]['Característica'].values)

        monetary_features = features[features['Monetario'] == 1]['Característica']

        # ### 1.3 Registro de las características de los datos en las estadísticas

        categorical_list = list(categorical_features)

        quantitative_list = list(quantitative_features)

        all_stats.base_stats.fields["Model"] = self.conf_column
        all_stats.base_stats.fields["Features"] = json.dumps({"Categorical":categorical_list,"Quantitative":quantitative_list})
        all_stats.base_stats.fields["Notes"] = self.notes
        return all_features, features_to_use, categorical_features, quantitative_features, monetary_features


    def prepare_data(self, all_stats, features_to_use, monetary_features):
        training_cps = None
        training_cps_aval = None

        source_fn, _ = os.path.splitext(self.source_file_name)
        processed_data_file_name = source_fn+self.conf_column+"Processed.csv"
        
        print('Data file: ', processed_data_file_name)

        data = get_storage().read_csv(
            f"{get_mlp_resource_key('processed_data_dir')}{processed_data_file_name}",
            parse_dates=["Fecha Inicial"],
            date_format="%Y-%m-%d",
            index_col=0
            )

        # Convertir a entero el índice
        data.index = data.index.astype(int)
        # Ordenar el DataFrame por el índice de menor a mayor
        data = data.sort_index()
        # pkdp.print_sizes(data)
        data_tracker = DataFrameEvolutionTracker(data)
        data_tracker.to_dataframe()

        # ### 2.1 Borramos la columna de 'Score' si está presente ya que es una variable 'Target' usada en otro contexto.

        if 'Score' in data.columns:
            data = data.drop(columns=['Score'])


        # ### 2.2 solo tomamos los datos hasta la fecha indicada

        print(data.shape)
        data = data[(data['Fecha Inicial'] <= self.max_date) & (data['Fecha Inicial'] >= self.min_date)]
        pkdp.print_sizes(data)
        data_tracker.register_step("Filtered by date range", data)
        data_tracker.to_dataframe()

        # ### 2.3 Registramos datos para las estadísticas

        all_stats.base_stats.fields["Size"] = len(data)

        # ## 4 Simplificación del CP a n cifras

        if self.zip_digits <= 4:
            print("Simplificando CP")
            if 'CP' in features_to_use:
                # Guardamos el conjunto de CP para almacenarlos posteriormente en 
                # el campo data_parameters de un modelo
                training_cps = data[['CP']]
                training_cps = training_cps.drop_duplicates().reset_index(drop=True)
            
                pkdp.transform_CP(data,'CP', self.zip_digits)
            
            if 'CP Aval' in features_to_use:
                # Guardamos el conjunto de CP Aval para almacenarlos posteriormente en 
                # el campo data_parameters de un modelo
                training_cps_aval = data[['CP Aval']]
                training_cps_aval = training_cps_aval.drop_duplicates().reset_index(drop=True)
                training_cps_aval = training_cps_aval.rename(columns={'CP Aval':'CP'})
            
                pkdp.transform_CP(data,'CP Aval', self.zip_digits)


        converter = pmdp.CurrentValueConverter(get_mlp_resource_key("inflation_table"))

        print("Converting monetary features to current value")
        for feature in monetary_features:
            data = converter.to_current_value(data, 'Fecha Inicial', feature)


        #data.to_excel("ProcessedData2.xlsx")
        print("Converting monetary features to multiple of 500")
        converter_500 = pmdp.MultipleAdjuster(500)
        data = converter_500.adjust_amount(data, 'Monto')


        # Monto promedio de los "buenos pagadores"
        good_payer = data[data['Incobrable']==0]

        # Monto promedio de los "malos pagadores"
        bad_payer = good_payer = data[data['Incobrable']==1]

        # Promedio de "Incobrabilidad"

        # Promedio del monto de préstamo
        return data, data_tracker, training_cps, training_cps_aval


    def build_preprocessor(self, data, categorical_features, quantitative_features):
        from sklearn.compose import make_column_transformer
        from sklearn.preprocessing import OrdinalEncoder

        predictors = data.drop(columns=['Incobrable', 'Fecha Inicial'])

        target = data['Incobrable']

        preprocessor = make_column_transformer(
            (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=99999999),
                list(categorical_features),
            ),
            (StandardScaler(), list(quantitative_features)),
        )

        return preprocessor


    def create_models_tester(self, data, all_stats):
        train_type = pk.BY_SIZE

        if train_type == pk.BY_SIZE:

        
            totals = data["Incobrable"].value_counts()
        
            train_init_date = data.iloc[0]['Fecha Inicial']
            train_end_date = data.iloc[-1]['Fecha Inicial']
        
            data = data.drop('Fecha Inicial', axis=1)
        
            all_stats.base_stats.fields['Paid'] = int(totals[0])
            all_stats.base_stats.fields['Unpaid'] = int(totals[1])
            all_stats.base_stats.fields['AccMin'] = float(max(totals[0],totals[1])/(totals[0]+totals[1])*100)
        
            all_stats.base_stats.fields["TestSize"] = self.test_size
            models_tester = pk.ModelsTester(data, self.models_to_test, 
                                            self.test_size, split=pk.BY_SIZE)
        
        elif train_type == pk.BY_INDEX:
            dates = data[['Fecha Inicial']]
            data = data.drop('Fecha Inicial', axis=1)
        
            data_size = data.shape[0]
            train_size = int(data_size/2)
            test_init = int(data_size*4/5)
            test_size = int(data_size/5)

            print("Data size: ", data_size)
            print("Train size: ", train_size)
            print("Test init: ", test_init)
            print("Test size: ", test_size)

            test_data = data.iloc[test_init:test_init+test_size]

            train_init_date = dates.loc[data.index[0],'Fecha Inicial']
            train_end_date = dates.loc[data.index[train_size],'Fecha Inicial']
            test_init_date = dates.loc[data.index[test_init],'Fecha Inicial']
            test_end_date = dates.loc[data.index[test_init+test_size-1],'Fecha Inicial']

            print("Evalueating with:")
            print(f"Train from: {train_init_date} to {train_end_date}")
            print(f"Test from: {test_init_date} to {test_end_date}")

            all_stats.base_stats.fields["TrainInitDate"] = str(train_init_date)
            all_stats.base_stats.fields["TrainEndDate"] = str(train_end_date)
            all_stats.base_stats.fields["TestInitDate"] = str(test_init_date)
            all_stats.base_stats.fields["TestEndDate" ] = str(test_end_date)

            all_stats.base_stats.fields["Size"] = len(test_data)

            totals = test_data["Incobrable"].value_counts()
            all_stats.base_stats.fields['Paid'] = int(totals[0])
            all_stats.base_stats.fields['Unpaid'] = int(totals[1])

            all_stats.base_stats.fields['AccMin'] = float(max(totals[0],totals[1])/(totals[0]+totals[1])*100)
        
            models_tester = pk.ModelsTester(data, self.models_to_test, self.test_size,
                                            train_init=0, train_size=train_size, test_init=test_init, 
                                            test_size=test_size, split=pk.BY_INDEX)
            all_stats.base_stats.fields["TestSize"] =  models_tester.x_test.shape[0]/models_tester.data.shape[0]
        else:
            train_init_date = data.iloc[0]['Fecha Inicial']
            train_end_date = pd.to_datetime('2022-12-22')

            test_init_date = train_end_date + pd.Timedelta(days=1)
            test_end_date = pd.to_datetime('2023-03-22')
        
            print("Evalueating with:")
            print(f"Train from: {train_init_date} to {train_end_date}")
            print(f"Test from: {test_init_date} to {test_end_date}")
        
    
            all_stats.base_stats.fields["TrainInitDate"] = str(train_init_date)
            all_stats.base_stats.fields["TrainEndDate"] = str(train_end_date)
            all_stats.base_stats.fields["TestInitDate"] = str(test_init_date)
            all_stats.base_stats.fields["TestEndDate" ] = str(test_end_date)

        
            models_tester = pk.ModelsTester(data, self.models_to_test,  
                                            train_end_date=train_end_date,
                                            test_end_date=test_end_date,
                                            split=pk.BY_DATE)
        
            all_stats.base_stats.fields["TestSize"] = models_tester.x_test.shape[0]/models_tester.data.shape[0]
            all_stats.base_stats.fields["Size"] = models_tester.data.shape[0]

            totals = models_tester.y_test.value_counts()
            all_stats.base_stats.fields['Paid'] = int(totals[0])
            all_stats.base_stats.fields['Unpaid'] = int(totals[1])

            all_stats.base_stats.fields['AccMin'] = float(max(totals[0],totals[1])/(totals[0]+totals[1])*100)

        return data, models_tester, train_init_date, train_end_date


    def train_and_evaluate(self, models_tester, preprocessor, all_stats, all_features, features_to_use, training_cps, training_cps_aval, train_init_date, train_end_date, str_date_and_time):
        models_tester.run_all_models(preprocessor)

        # ### Correr los modelos con los datos de prueba y desplegar las estadísticas

        for model in models_tester.models:
            print(f"********************* Results for model: { model.name} : {model.parameters} *********************")
            print("********************** TEST **********************")
            y_pred_test = model.pipeline.predict(models_tester.x_test)
            all_stats.add_and_print_stats(models_tester.y_test, y_pred_test, model.name, model.parameters, "Test")

            print("********************** TRAIN **********************")
            y_pred_train = model.pipeline.predict(models_tester.x_train)
            all_stats.add_and_print_stats(models_tester.y_train, y_pred_train, model.name, model.parameters, "Train")
        
            #if (model.name != "Random Forest") and (model.name != "Ada Boost"):
            if (model.name != "Ada Boost"):
                ### Estadisticas con el mejor umbral para el accurracy y monetary gain

                predictions = model.pipeline.predict_proba(models_tester.x_test)[:,1]
                acceptance_threshold, accurracy_list, best_threshold, best_accurracy, gl, bgt, bg =  \
                                                                                pk.find_best_threshold(predictions, models_tester.y_test)
                model.best_gain_threshold = bgt
                model.best_accurracy_threshold = best_threshold
            
                print("**************** Unbral para best accurracy *******************")
                print("Best threshold = ", best_threshold)
                pk.plot_threshold_evolution(acceptance_threshold, accurracy_list, "Accurracy of the model by changing the threshold",
                                            "Threshold","Accuracy",
                                            f"{self.conf_column}_{model.name}_accuracy_test.png", show=False)
                y_pred_accuracy = np.where(predictions > best_threshold, 1, 0)
                all_stats.add_and_print_stats(models_tester.y_test, y_pred_accuracy, model.name, model.parameters, 
                                            "Best accurracy threshold Test", True,
                                            f"CM_{self.conf_column}_{model.name}_accuracy_test.png")

                ### Estadísticas de con la mejor ganancia monetária
                print("**************** Unbral para best monetary gain Test *******************")
                print("Best gain threshold = ", bgt)
                pk.plot_threshold_evolution(acceptance_threshold, gl, 
                                            "Monetary gain of the model by changing the threshold (Test)", 
                                            "Threshold", "Monetary gain",
                                            f"{self.conf_column}_{model.name}_best_gain_test.png", show=False)
                y_pred_monetary_test = np.where(predictions > bgt, 1, 0)
                print(f"Model configuration:{self.conf_column}")
                all_stats.add_and_print_stats(models_tester.y_test, y_pred_monetary_test, model.name, model.parameters, 
                                            "Best monetary gain threshold Test", True,
                                            f"CM_{self.conf_column}_{model.name}_monetary_test.png")

                ### Estadísticas con la mejor ganancia monetária datos de entrenamiento
                print("**************** Unbral para best monetary gain Train *******************")
                predictions = model.pipeline.predict_proba(models_tester.x_train)[:,1]
                acceptance_threshold, accurracy_list, best_threshold, best_accurracy, gl, bgt, bg = pk.find_best_threshold(predictions, models_tester.y_train)

                print("Best gain threshold = ", bgt, "with a monetary gain of", bg)
                pk.plot_threshold_evolution(acceptance_threshold, gl, "Monetary gain of the model by changing the threshold (Train)",
                                            "Threshold", "Monetary gain",
                                            f"{self.conf_column}_{model.name}_best_gain_train.png", show=False)

                y_pred_monetary_train = np.where(predictions > bgt, 1, 0)
                all_stats.add_and_print_stats(models_tester.y_train, y_pred_monetary_train, model.name, model.parameters,"Best monetary gain threshold Train")
            #model.min_max_stats = min_max_stats
            model.columns = models_tester.x_train.columns
            model.feature_engineering = self.feature_engineering
            model.features = all_features
            model.conf_column = self.conf_column
            model.data_parameters['zip_digits'] = self.zip_digits
            model.data_parameters['Training init date'] = train_init_date
            model.data_parameters['Trainig en date'] = train_end_date
            model.data_parameters['SourceFile'] = self.source_file_name
            model.data_parameters['Date'] = str_date_and_time
            if 'CP' in features_to_use:
                model.data_parameters['CP'] = training_cps
            if 'CP Aval' in features_to_use:
                model.data_parameters['CP Aval'] = training_cps_aval
            model.save_with_pickle(get_storage())
            plt.close("all")

        for model in models_tester.models:
            print(model)
            print("=====================================")
        return y_pred_monetary_test


    def export_statistics_and_predictions(self, all_stats, data_tracker, models_tester, preprocessor, y_pred_monetary_test):
        storage = get_storage()
        con = sqlite3.connect(BASE_DIR / "promocash_stats.db")
        cur = con.cursor()
        cur.executemany("INSERT INTO stats VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", all_stats.values)
        con.commit()

        # ### Leer de la base de datos para escribir las estadísticas en la base de datos

        df = pd.read_sql_query("SELECT * FROM stats", con)

        time_now = datetime.now()

        format_date_hour = "%Y-%m-%d %H:%M:%S"
        date_and_hour = time_now.strftime(format_date_hour)

        con.commit()
        con.close()

        df.index.name = 'index'
        storage.write_excel(f"{get_mlp_resource_key('stats_dir')}{self.stats_file_name}", df)

        # Escribir el archivo de data tracking
        outputs_dir = get_mlp_resource_key("outputs_dir")

        storage.write_csv(
            f"{outputs_dir}{self.conf_column}_data_tracking_train.csv",
            data_tracker.to_dataframe(),
            index=False
        )

        # data_tracker.to_dataframe().to_csv("./data/" + self.conf_column + "_data_tracking_train.csv")

        test_result = models_tester.x_test.copy()
        test_result["Prediction"] = y_pred_monetary_test
        test_result["Incobrable"] = models_tester.y_test

        storage.write_csv(
            f"{outputs_dir}TestResult_{self.conf_column}.csv",
            test_result,
            index=False
        )

        # test_result.to_csv("./data/TestResult"+self.conf_column+".csv")

        preprocessor.fit(models_tester.x_train)
        x_test_processed = preprocessor.transform(models_tester.x_test)

        # Crear DataFrame a partir de los datos transformados
        x_test_transformed_df = pd.DataFrame(
            data=x_test_processed,
            columns=models_tester.x_test.columns
        )



        x_test_transformed_df["Prediccion"] = models_tester.y_test.to_numpy()



        time_now = datetime.now()

        date_and_hour = time_now.strftime(format_date_hour)

        # Imprimir la fecha y hora
        print("Ended at:", date_and_hour)

        # Hacer un sonido
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1000, 500)  # Frecuencia de 1000 Hz, duración de 500 ms
            winsound.Beep(500, 500) 
            winsound.Beep(1000, 500) 
        else:
            print('\a')  # Hace un beep en algunos sistemas UNIX
        
        print('Configuration : ', self.conf_column)

        len(x_test_transformed_df)

        return x_test_transformed_df

    def run(self):
        print(sys.executable)
        print("pandas", pd.__version__)
        print("numpy", np.__version__)
        print("sklearn", sklearn.__version__)

        date_and_time = datetime.now()
        str_date_and_time = date_and_time.strftime("%Y:%m:%d:%H:%M")
        print(str_date_and_time)
        all_stats = pk.ModelsStats()

        (all_features, features_to_use, categorical_features,
        quantitative_features, monetary_features) = self.load_configuration(all_stats)
        
        data, data_tracker, training_cps, training_cps_aval = self.prepare_data(
            all_stats, features_to_use, monetary_features
        )

        preprocessor = self.build_preprocessor(data, categorical_features, quantitative_features)

        data, models_tester, train_init_date, train_end_date = self.create_models_tester(data, all_stats)

        y_pred_monetary_test = self.train_and_evaluate(
            models_tester, preprocessor, all_stats, all_features, features_to_use,
            training_cps, training_cps_aval, train_init_date, train_end_date, str_date_and_time
        )

        x_test_transformed_df = self.export_statistics_and_predictions(
            all_stats, data_tracker, models_tester, preprocessor, y_pred_monetary_test
        )


