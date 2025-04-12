# backend/services/data_loader.py
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path  # Corrige esto (elimina self.backend.file_path)

    def load_data(self):
        """Returns two lists of lists"""
            # Leer encabezados y datos por separado
        df_info = pd.read_csv(self.file_path, nrows=3, header=None)
        df_data = pd.read_csv(self.file_path, skiprows=4, header=None)

        # Extraer información de planta y pozos
        field_name = df_info.iloc[2, 0] 
        well_names = df_info.iloc[2, 1:].dropna().tolist()  
        column_labels_qgl = df_data.columns[1::2]
        column_label_prod = df_data.columns[2::2]
        list_of_wells_qgl = df_data.loc[:, column_labels_qgl].T.to_numpy().tolist()
        list_of_well_prods = df_data.loc[:, column_label_prod].T.to_numpy().tolist()
        list_of_wells_qgl = [[x for x in q_gl if not np.isnan(x)] for q_gl in list_of_wells_qgl]
        list_of_well_prods = [[x for x in q_oil if not np.isnan(x)] for q_oil in list_of_well_prods]
        return list_of_wells_qgl, list_of_well_prods, [field_name] + well_names
    

if __name__ == "__main__":
    load = DataLoader('./data/gl_nishikiori_data_five_version_01.csv')
    var1 = load.load_data()
    print(var1[2])  # Corregido el índice
