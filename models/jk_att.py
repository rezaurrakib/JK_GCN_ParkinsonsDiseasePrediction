from gcn.models import *

flags = tf.app.flags
FLAGS = flags.FLAGS


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
        self.accuracy = masked_accuracy(self.outputs, self.placeholder['labels'], self.placeholder['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholder['labels'], self.placeholder['labels_mask'])

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


class JK_Att(JK_GCN):

    def __init__(self, placeholder, input_dim, depth, **kwargs):
        self.depth = depth
        self.shortcuts = []
        super(JK_Att, self).__init__(placeholder, input_dim, **kwargs)

    def build(self):
        """ Wrapper for _build() """
        print("Build call for Parallel Attention ......")
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        for index in range(len(self.layers) - 1):
            hidden = self.layers[index](self.activations[-1])
            self.activations.append(hidden)

        m1 = uniform([324, 50], name="attention_1")
        self.vars["attention_1"] = m1

        m2 = uniform([324, 50], name="attention_2")
        self.vars["attention_2"] = m2

        m3 = uniform([324, 50], name="attention_3")
        self.vars["attention_1"] = m3

        m4 = uniform([324, 50], name="attention_4")
        self.vars["attention_4"] = m4

        mult_stack = tf.stack([m1, m2, m3, m4], axis=1)
        self.attention_scores = tf.nn.softmax(mult_stack)
        hidden_layers = tf.stack([self.activations[1], self.activations[2], self.activations[3], self.activations[4]],
                                 axis=1)
        mult = tf.multiply(self.attention_scores, hidden_layers)
        sum = tf.reduce_sum(mult, 1)

        self.outputs = self.layers[-1](sum)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholder,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # For the time being, this is hard coded with 4 hidden layers.
        # But it will be taken as a variable length hidden layer depth
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
