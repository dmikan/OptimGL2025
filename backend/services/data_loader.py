# backend/services/data_loader.py
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path  # Corrige esto (elimina self.backend.file_path)

    def load_data(self):
        """Returns two lists"""
        df = pd.read_csv(self.file_path, sep=',')
        qgl = df['q_gl'].tolist()
        q_fluid_wells = df.iloc[:, 1:].T.to_numpy().tolist()
        return qgl, q_fluid_wells

    def load_data_gl_template(self):
        """Returns two lists of lists"""
        df = pd.read_csv(self.file_path, sep=',')
        column_labels_qgl = df.columns[1::2]
        column_label_prod = df.columns[2::2]
        list_of_wells_qgl = df.loc[:, column_labels_qgl].T.to_numpy().tolist()
        list_of_well_prods = df.loc[:, column_label_prod].T.to_numpy().tolist()
        list_of_wells_qgl = [[x for x in q_gl if not np.isnan(x)] for q_gl in list_of_wells_qgl]
        list_of_well_prods = [[x for x in q_oil if not np.isnan(x)] for q_oil in list_of_well_prods]
        return list_of_wells_qgl, list_of_well_prods