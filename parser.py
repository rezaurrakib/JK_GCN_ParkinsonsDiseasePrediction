import os
import csv
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

root_folder = os.getcwd()
data_folder = os.path.join(root_folder, "Dataset/non_imaging_ppmi_data.xls")
patient_ids_file = os.path.join(root_folder, "Dataset/ppmi_labels.csv")
xls = pd.ExcelFile(data_folder)

class AffinityGraph():
    """
        Class for Graph Creation. Each Node represents a patient.
        Edge connections are based on MOCA & UPDRS experiment results of each patient.
        Phenotype information (i.e., Age and Gender) has also been used.
    """
    def __init__(self):
        self.updrs = pd.read_excel(xls, "UPDRS")
        self.moca = pd.read_excel(xls, 'MOCA')
        self.gender_age = pd.read_excel(xls, 'Gender_and_Age')
        self.weight = pd.read_excel(xls, 'Weight')

    def get_embedded_patient_ids(self):
        patient_ids = []
        file_reader = open(patient_ids_file, "rt")
        reader = csv.reader(file_reader)
        for row in reader:
            patient_ids.append(int(float(row[0])))
        return patient_ids

    def get_embedded_patients_classes(self):
        patients_classes = []
        file_reader = open(patient_ids_file, "rt")
        reader = csv.reader(file_reader)
        for row in reader:
            patients_classes.append(int(float(row[1])))
        return patients_classes

    # Get MCATOT values for patients from MOCA Sheet
    def get_moca_score(self, patient_ids):
        """
        return:
            scores_dict : MOCA values for all patient
        """
        scores_dict = {}
        for key in patient_ids:
            scores_dict[key] = None

        for idx, row in self.moca.iterrows():
            if row[2] in scores_dict:  # Means, this row has valid patient id
                if scores_dict[row[2]] == None:
                    if row[3] == 'V01':
                        scores_dict[row[2]] = int(row[33])
                    elif row[3] == 'SC':
                        scores_dict[row[2]] = int(row[33])

        return scores_dict

    # Get NP3*** values for a list of patients from UPDRS Sheet
    def get_updrs_score(self, patient_ids):
        """
        return:
            scores_dict : updrs values for all patient
        """
        scores_dict = {}
        for key in patient_ids:
            scores_dict[key] = None

        for idx, row in self.updrs.iterrows():
            if row[2] in scores_dict:  # Means, this row has valid patient id
                if scores_dict[row[2]] == None:
                    # Sum from NP3SPCH upto NP3RTCON (col 8 - col 40 inclusive)
                    if row[3] == 'BL':
                        sum = 0
                        for i in range(8, 41):
                            if pd.isnull(self.updrs.iloc[idx, i]) == False:
                                sum += row[i]
                        scores_dict[row[2]] = int(sum)

                    elif row[3] == 'V01':
                        sum = 0
                        for i in range(8, 41):
                            if pd.isnull(self.updrs.iloc[idx, i]) == False:
                                sum += row[i]
                        scores_dict[row[2]] = int(sum)

        return scores_dict

    # Get Gender for patients from Gender_and_Age Sheet
    def get_gender(self, patient_ids):
        """
        return:
            genders_dict : Identify gender for all patient
        """
        genders_dict = {}
        for key in patient_ids:
            genders_dict[key] = None

        for idx, row in self.gender_age.iterrows():
            if row[2] in genders_dict:  # Means, this row has valid patient id
                if genders_dict[row[2]] == None:
                    genders_dict[row[2]] = int(row[11])

        return genders_dict

    # Calculate Age for patients from Gender_and_Age Sheet
    def get_age(self, patient_ids):
        """
        return:
            ages_dict : Calculate age for all patient
        """
        ages_dict = {}
        for key in patient_ids:
            ages_dict[key] = None

        for idx, row in self.gender_age.iterrows():
            if row[2] in ages_dict:  # Means, this row has valid patient id
                if ages_dict[row[2]] == None:
                    ages_dict[row[2]] = int(2018 - row[10])

        return ages_dict

    def creating_adjacency_matrix_updrs(self, pat_ids, updrs_dict):
        num_nodes = len(pat_ids)
        graph = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if updrs_dict[pat_ids[i]] == updrs_dict[pat_ids[j]]:
                    graph[i, j] = 1
                    graph[j, i] = 1

        return graph

    def creating_adjacency_matrix_moca(self, pat_ids, moca_dict):
        num_nodes = len(pat_ids)
        graph = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if moca_dict[pat_ids[i]] == moca_dict[pat_ids[j]]:
                    graph[i, j] = 1
                    graph[j, i] = 1

        return graph

    def creating_adjacency_matrix_age(self, pat_ids, ages_dict):
        num_nodes = len(pat_ids)
        graph = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if abs(ages_dict[pat_ids[i]] - ages_dict[pat_ids[j]]) <= 2:
                    graph[i, j] = 1
                    graph[j, i] = 1

        return graph

    def creating_adjacency_matrix_gender(self, pat_ids, gender_dict):
        num_nodes = len(pat_ids)
        graph = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if gender_dict[pat_ids[i]] == gender_dict[pat_ids[j]]:
                    graph[i, j] = 1
                    graph[j, i] = 1

        return graph

    def creating_final_graph(self, pat_ids, updrs_graph, moca_graph, age_graph, gender_graph):
        num_nodes = len(pat_ids)
        f_graph = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    f_graph[i, j] = 1

                # Considering all information
                f_graph[i, j] += updrs_graph[i, j] + moca_graph[i, j] + age_graph[i, j] + gender_graph[i, j]

                # Only MOCA information
                #f_graph[i, j] += moca_graph[i, j]

                # Only UPDRS information
                #f_graph[i, j] += updrs_graph[i, j]

                # Only Age information
                #f_graph[i, j] += age_graph[i, j]

                # Only Gender information
                #f_graph[i, j] += gender_graph[i, j]


        return f_graph

    def person_corr_calc(self, moca, updrs, age, gender, patient_ids, labels):
        data = pd.DataFrame()
        moca_vec = []
        updrs_vec = []
        age_vec = []
        gender_vec = []
        print(len(patient_ids))
        print(len(moca_vec))
        for x in patient_ids:
            #print(moca[x])
            moca_vec.append(moca[x])
            updrs_vec.append(updrs[x])
            age_vec.append(age[x])
            gender_vec.append(gender[x])

        print(len(moca_vec))
        print(len(updrs_vec))

        data['moca'] = moca_vec
        data['updrs'] = updrs_vec
        data['gender'] = gender_vec
        data['age'] = age_vec
        data['label'] = labels


        print(data.corr(method='pearson'))
        print(data.corr(method='spearman'))


if __name__ == "__main__":
    graph_obj = AffinityGraph()
    patient_ids = graph_obj.get_embedded_patient_ids()
    patients_classes = graph_obj.get_embedded_patients_classes()
    updrs_dict = graph_obj.get_updrs_score(patient_ids)
    moca_dict = graph_obj.get_moca_score(patient_ids)
    ages_dict = graph_obj.get_age(patient_ids)
    gender_dict = graph_obj.get_gender(patient_ids)

    np.set_printoptions(threshold=np.nan)  # For printing full Numpy array

    updrs_graph = graph_obj.creating_adjacency_matrix_updrs(patient_ids, updrs_dict)
    moca_graph = graph_obj.creating_adjacency_matrix_moca(patient_ids, moca_dict)
    age_graph = graph_obj.creating_adjacency_matrix_age(patient_ids, ages_dict)
    gender_graph = graph_obj.creating_adjacency_matrix_gender(patient_ids, gender_dict)
    final_adj_matrix = graph_obj.creating_final_graph(patient_ids, updrs_graph, moca_graph, age_graph, gender_graph)

    print(final_adj_matrix.shape)
