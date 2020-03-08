from __future__ import division
from __future__ import print_function

import time
import random
import datetime
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt

from gcn.utils import *
from operator import add
from Jumping_Knowledge_Network.models.jk_att import *
from Jumping_Knowledge_Network.visualization import *
from Jumping_Knowledge_Network.models.jk_lstm import *
from Jumping_Knowledge_Network.models.jk_concat import *
from Jumping_Knowledge_Network.models.jk_maxpool import *


def get_train_test_masks(labels, idx_train, idx_val, idx_test, all_label):
    all_label = all_label - 1
    lbl = all_label[0:]

    num_nodes = 324
    class_a = [i for i in range(324) if lbl[i] == 0]
    class_b = [i for i in range(324) if lbl[i] == 1]

    freq_a = len(class_a)
    freq_b = len(class_b)
    print("Class A: ", freq_a)
    print("Class B: ", freq_b)

    # Initializing weights for weighted cross entropy
    node_weights = np.zeros((324,))
    node_weights[class_a] = 1 - freq_a / float(num_nodes)
    node_weights[class_b] = 1 - freq_b / float(num_nodes)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print("train mask shape: ", train_mask.shape)
    print("train mask data: ", train_mask)

    print("val mask shape: ", val_mask.shape)
    print("val mask data: ", val_mask)

    print("test mask shape: ", test_mask.shape)
    print("test mask data: ", test_mask)

    # changing mask for having weighted loss
    train_mask = node_weights * train_mask
    val_mask = node_weights * val_mask
    test_mask = node_weights * test_mask
    return y_train, y_val, y_test, train_mask, val_mask, test_mask


# Create Support and Placeholders for different K order Cheby Polynomial value
# For the time being, not needed
# @Author : Reza

def preparing_support_and_placeholders(k_values, adj, features, y_train):
    support_list = []
    placeholder_list = []
    for i in range(len(k_values)):
        support = chebyshev_polynomials(adj, k_values[i])
        num_supports = k_values[i] + 1
        support_list.append(support)

        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'phase_train': tf.placeholder_with_default(False, shape=()),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }
        placeholder_list.append(placeholders)

    return support_list, placeholder_list


# Create feed_dict for JKNetworks with different K values
# For the time being, not needed
# @Author : Reza

def jknetwork_construct_feed_dict(features, support_list, labels, labels_mask, placeholder_list):
    """Construct feed dictionary."""
    feed_dict = dict()
    for index in range(4):
        feed_dict.update({placeholder_list[index]['labels']: labels})
        feed_dict.update({placeholder_list[index]['labels_mask']: labels_mask})
        feed_dict.update({placeholder_list[index]['features']: features})
        feed_dict.update({placeholder_list[index]['support'][i]: support_list[index][i] for i in range(len(support_list[index]))})
        feed_dict.update({placeholder_list[index]['num_features_nonzero']: features[1].shape})
    return feed_dict


def create_flagsparams(params):
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')


def run_training(adj, features, labels, idx_train, idx_val, idx_test, params, train_acc_val, test_acc_val,
                 train_loss_val, test_loss_val, all_label, fold):
    # Set random seed
    tf.reset_default_graph()
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    tf.set_random_seed(params['seed'])

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test, all_label)

    # Some pre processing
    features = preprocess_features(features)
    print("Feature Dimension is : ", features[2][0])

    # For JK Network using fixed placeholder
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree

    # Create JK Model
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # JK Network With Concatenation
    #jk_model_func = JK_CONCAT
    #jk_model = jk_model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True)

    # JK Network With Max-Pooling
    # jk_model_func = JK_MAXPOOL
    # jk_model = jk_model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True)

    # JK Network With LSTM
    jk_model_func = JK_LSTM
    jk_model = jk_model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True)

    # JK Network With Parallel Attention
    #jk_model_func = JK_Att
    #jk_model = jk_model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True)

    # Initialize session
    #sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Define model evaluation function
    def evaluate(feats, graph_list, label, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph_list, label, mask, placeholders)
        feed_dict_val.update({placeholders['phase_train'].name: False})

        # Validation for JK Network using GCN as Base Model
        outs_val = sess.run([jk_model.loss, jk_model.accuracy, jk_model.predict(), merged_summary], feed_dict=feed_dict_val)
        lab = label
        return outs_val[0], outs_val[1], outs_val[2], outs_val[3],  (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    train_acc_val_x = []
    test_acc_val_x = []
    train_loss_val_x = []
    test_loss_val_x = []

    feed_dict = ()

    # defining train, test and val writers in /tmp/model_name/ path
    path = os.getcwd()
    file_writer = "FileWriter"
    if not os.path.exists(file_writer):
        os.makedirs(file_writer)

    log_dir = os.path.join(path, file_writer + '/' + jk_model.name)
    train_writer = tf.summary.FileWriter(logdir=log_dir + '/train_fold_{}/'.format(fold + 1))
    test_writer = tf.summary.FileWriter(logdir=log_dir + '/test_fold_{}/'.format(fold + 1))
    val_writer = tf.summary.FileWriter(logdir=log_dir + '/val_fold_{}/'.format(fold + 1))

    # loss and accuracy scalar curves for Tensorboard
    tf.summary.scalar(name='loss_fold_{}'.format(fold + 1), tensor=jk_model.loss)
    tf.summary.scalar(name='accuracy_fold_{}'.format(fold + 1), tensor=jk_model.accuracy)
    merged_summary = tf.summary.merge_all()


    # Train model
    for epoch in range(params['epochs']):
        t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout, placeholders['phase_train']: True})

        train_results = sess.run([jk_model.opt_op, jk_model.loss, jk_model.accuracy, jk_model.predict(), merged_summary], feed_dict=feed_dict)
        train_writer.add_summary(train_results[-1], epoch)

        # Evaluation on val set
        val_loss, val_acc, pred_label, val_summary, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(val_loss)
        val_writer.add_summary(val_summary, epoch)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_results[1]),
              "train_acc=", "{:.5f}".format(train_results[2]), "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

        train_acc_val_x.append(train_results[2])
        test_acc_val_x.append(val_acc)
        train_loss_val_x.append(train_results[1])
        test_loss_val_x.append(val_loss)


    if not train_acc_val:
        print(" ---------- At First Iteration ----------")
        train_loss_val = train_loss_val_x
        test_loss_val = test_loss_val_x
        train_acc_val = train_acc_val_x
        test_acc_val = test_acc_val_x
    else:
        train_loss_val = list(map(add, train_loss_val, train_loss_val_x))
        test_loss_val = list(map(add, test_loss_val, test_loss_val_x))
        train_acc_val = list(map(add, train_acc_val, train_acc_val_x))
        test_acc_val = list(map(add, test_acc_val, test_acc_val_x))

    print("Model Name: ", jk_model.name)

    # Visualizing layers' embedding
    hidden_1, hidden_2, hidden_3, hidden_4, final_layer = sess.run([jk_model.activations[1], jk_model.activations[2],
                                                                    jk_model.activations[3], jk_model.activations[4],
                                                                    jk_model.outputs], feed_dict=feed_dict)

    features_embedding_visualize_hidden_layer(final_layer, all_label, fold, jk_model.name)

    # Testing the Network
    sess.run(tf.local_variables_initializer())
    test_cost, test_acc, test_pred, test_summary, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    test_writer.add_summary(test_summary, epoch)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc))

    # Closing writers
    train_writer.close()
    test_writer.close()
    val_writer.close()

    return test_acc, train_loss_val, test_loss_val, train_acc_val, test_acc_val
