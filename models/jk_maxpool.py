from gcn.models import *

flags = tf.app.flags
FLAGS = flags.FLAGS


## Base model : GCN   ##
## JKNetwork Model    ##
## @Author : Reza     ##


class JK_GCN(Model):
    def __init__(self, placeholder, input_dim, **kwargs):
        super(JK_GCN, self).__init__(**kwargs)

        self.inputs = placeholder['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholder['labels'].get_shape().as_list()[1]
        self.placeholder = placeholder

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholder['labels'],
                                                  self.placeholder['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholder['labels'],
                                        self.placeholder['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholder['labels'],
                              self.placeholder['labels_mask'])

    def _build(self):
        print("GCN Build call")
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)



class JK_MAXPOOL(JK_GCN):
    def __init__(self, placeholder, input_dim, depth, **kwargs):
        self.depth = depth
        self.shortcuts = []
        super(JK_MAXPOOL, self).__init__(placeholder, input_dim, **kwargs)

    def build(self):
        """ Wrapper for _build() """
        print("Method call for concat ......")
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        for index in range(len(self.layers) - 1):
            hidden = self.layers[index](self.activations[-1])
            self.activations.append(hidden)

        self.outputs = tf.stack([self.activations[1], self.activations[2], self.activations[3], self.activations[4]], axis=1)

        # apply max-out over different GC layers
        self.outputs = tf.nn.pool(self.outputs, [4], pooling_type='MAX', padding='VALID', data_format='NWC')
        self.outputs = tf.reshape(self.outputs, [324, FLAGS.hidden1])
        self.outputs = self.layers[-1](self.outputs)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        # Hidden Layer of Size: 4
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholder,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(3):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholder,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholder,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
