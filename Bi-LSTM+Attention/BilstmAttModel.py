import tensorflow as tf

# 构建模型
class BiLSTMAttention(object):
    def __init__(self, config, wordEmbedding):

        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, config.numClasses], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))
                    self.embeddedWords = tf.concat(outputs_, 2)

        outputs = tf.split(self.embeddedWords, 2, -1)

        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]
            output = self.attention(H, config)
            outputSize = config.model.hiddenSizes[-1]

        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")

            one = tf.ones_like(self.logits)
            zero = tf.zeros_like(self.logits)
            self.predictions = tf.where(self.logits < 0.5, x=zero, y=one, name='predictions')  # 0.5为阈值

        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputY, logits=self.logits)
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def attention(self, H, config):
        hiddenSize = config.model.hiddenSizes[-1]
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        M = tf.tanh(H)
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])
        self.alpha = tf.nn.softmax(restoreM)
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        sentenceRepren = tf.tanh(sequeezeR)
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        return output