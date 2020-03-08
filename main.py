import os
import time
import datetime
import argparse
import xlsxwriter
import numpy as np
import scipy.io as sio
import sklearn.metrics
import matplotlib.pyplot as plt

from scipy import sparse
#from joblib import Parallel, delayed
from sklearn.manifold import TSNE
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

import Jumping_Knowledge_Network.parser as Reader
import Jumping_Knowledge_Network.training as train


root_folder = os.getcwd()
file_dim_reduction = os.path.join(root_folder, "Dataset/dim_reduction_output.txt")
file_corr_mat = os.path.join(root_folder, "Dataset/correlation_matrix.txt")
fig_folder = os.path.join(root_folder, "Visualization/")


def create_flags(params):
    train.create_flagsparams(params)


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(train_ind, test_ind, val_ind, graph_feat, features, correlation_matrix, y, params, patients_ids, train_acc_val, test_acc_val,
               train_loss_val, test_loss_val, fold):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotype measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        params          : dictionary of GCNs parameters
    returns:
        test_acc       : average accuracy over the test samples using GCNs
        lin_acc        : average accuracy over the test samples using the linear classifier
        lin_auc        : average area under curve over the test samples using the linear classifier
        fold_size      : number of test samples
        train_loss_val : Sum of 10 Folds training loss values
        test_loss_val  : Sum of 10 Folds testing loss values
        train_acc_val  : Sum of 10 Folds training accuracy values
        test_acc_val   : Sum of 10 Folds testing accuracy values
    """

    # selection of a subset of data if running experiments with a subset of the training setl
    y = np.array(y)
    features = np.array(features)
    x_data = features[train_ind, :]
    y_data = np.transpose(y[train_ind]) - 1
    y_data_2 = np.transpose(y) - 1
    y_data_x = np.vstack((y_data_2, abs(y_data_2 - 1))).T
    fold_size = len(test_ind)

    # Calculate all pairwise distances
    dist = correlation_matrix
    sigma = np.mean(dist)

    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph

    # Linear classifier
    clf = RidgeClassifier()
    clf.fit(x_data, y_data)

    # Compute the accuracy
    lin_acc = clf.score(x_data, y_data)
    print("Linear Accuracy: ", lin_acc)
    print("Test Indices: ", test_ind)

    # Compute the AUC
    pred = clf.decision_function(x_data)
    lin_auc = sklearn.metrics.roc_auc_score(y_data, pred)

    # Classification with GCNs
    test_acc, train_loss_val, test_loss_val, train_acc_val, test_acc_val = train.run_training(final_graph,
            sparse.coo_matrix( np.array(features)).tolil(), y_data_x, train_ind, val_ind, test_ind, params,
            train_acc_val, test_acc_val, train_loss_val, test_loss_val, y, fold)

    # return number of correctly classified samples instead of percentage
    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))



    return test_acc, lin_acc, lin_auc, fold_size, train_loss_val, test_loss_val, train_acc_val, test_acc_val


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='JK Network for population graphs: ' 
                                                 'classification of the PPMI dataset')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float, help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=50, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.001, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                                                      'for more options )')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                       'uses chebyshev polynomials, '
                                                       'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=10, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    args = parser.parse_args()

    # GCN Parameters
    params = dict()
    params['model'] = args.model  # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate  # Initial learning rate
    params['epochs'] = args.epochs  # Number of epochs to train
    params['dropout'] = args.dropout  # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden  # Number of units in hidden layers
    params['decay'] = args.decay  # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params['epochs']  # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3  # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth  # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed  # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features  # number of features for feature selection step
    params['num_training'] = args.num_training  # percentage of training set used for training

    # Graph
    graph_obj = Reader.AffinityGraph()
    patient_ids = graph_obj.get_embedded_patient_ids()
    patients_classes = graph_obj.get_embedded_patients_classes()
    num_nodes = len(patient_ids)

    updrs_dict = graph_obj.get_updrs_score(patient_ids)
    moca_dict = graph_obj.get_moca_score(patient_ids)
    ages_dict = graph_obj.get_age(patient_ids)
    gender_dict = graph_obj.get_gender(patient_ids)

    updrs_graph = graph_obj.creating_adjacency_matrix_updrs(patient_ids, updrs_dict)
    moca_graph = graph_obj.creating_adjacency_matrix_moca(patient_ids, moca_dict)
    age_graph = graph_obj.creating_adjacency_matrix_age(patient_ids, ages_dict)
    gender_graph = graph_obj.creating_adjacency_matrix_gender(patient_ids, gender_dict)

    graph = graph_obj.creating_final_graph(patient_ids, updrs_graph, moca_graph, age_graph, gender_graph)

    # Get Features
    features = np.genfromtxt(file_dim_reduction, dtype='float', delimiter=',')

    # Get Correlation Matrix
    correlation_matrix = np.genfromtxt(file_corr_mat, dtype='float', delimiter=',')

    # Initialise variables for class labels and acquisition sites
    y = patients_classes

    cnt1 = cnt2 = 0
    for i in range(len(y)):
        if(y[i] == 1):
            cnt1 += 1
        else:
            cnt2 += 1

    print("Class 1: ", cnt1, " Class 2: ", cnt2)

    print("---------------- Training Started --------------")
    train_acc_val = []
    test_acc_val = []
    train_loss_val = []
    test_loss_val = []

    # Flag Creation
    create_flags(params)
    print("arg.folds = ", args.folds)
    # Folds for cross validation experiments
    num_folds = args.folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Running 10 folds
    for fold in range(0, num_folds):
        cv_splits = list(skf.split(features, np.squeeze(y)))
        train = cv_splits[fold][0]
        test = cv_splits[fold][1]
        val = test
        scores_acc, scores_lin, scores_auc_lin, fold_size, train_loss_val, test_loss_val, train_acc_val, test_acc_val = train_fold(
            train, test, val, graph, features, correlation_matrix, y, params, patient_ids, train_acc_val, test_acc_val,
            train_loss_val, test_loss_val, fold)

    test_loss_val = [x / num_folds for x in test_loss_val]
    train_loss_val = [x / num_folds for x in train_loss_val]
    test_acc_val = [x / num_folds for x in test_acc_val]
    train_acc_val = [x / num_folds for x in train_acc_val]

    print("Results:", "train loss=", "{:.5f}".format(train_loss_val[-1]), "train accuracy=", "{:.5f}".format(train_acc_val[-1]), "test loss=", "{:.5f}".format(test_loss_val[-1]),
          "test accuracy=", "{:.5f}".format(test_acc_val[-1]))


if __name__ == "__main__":
    main()
