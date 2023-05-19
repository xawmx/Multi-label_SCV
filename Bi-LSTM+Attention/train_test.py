import os
import shutil

from BilstmAttModel import BiLSTMAttention
import datetime
import json
import warnings
from collections import Counter
import gensim
import numpy as np
import tensorflow as tf
from evaluate import get_binary_metrics, mean
from multiEvl import multiPartEvl
warnings.filterwarnings("ignore")

#查看是否使用GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# 配置参数
class TrainingConfig(object):
    epoches = 4
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    hiddenSizes = [256, 128]  # LSTM结构的神经元个数
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 500
    dataSource = "../sc_data/bertData/keycentence/pre_training_data.json"
    testDataSource = "../sc_data/bertData/keycentence/pre_training_data.json"

    batchSize = 64
    stopWordSource = "../sc_data/english"
    numClasses = 6
    rate = 0.8
    training = TrainingConfig()
    model = ModelConfig()

config = Config()

typeTrain = "train"
if config.dataSource.__contains__("Key"):
    typeTrain = "pretrainKey"
logFile = open(typeTrain + "log" + str(config.sequenceLength) + ".txt", "w")
logFile.write(str(config.sequenceLength) + "\n")
logFile.write(str(config.dataSource) + "\n")

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        self._stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.wordEmbedding = None
        self.labelList = []

    def _readData(self, filePath):
        with open(filePath, "r", encoding='utf-8') as file:
            df = json.load(file)
            labels = []
            review = []
            i = 0
            for line in df:
                if i >= 7600:
                    # 训练集和测试集 比例 8：2
                    break
                i += 1
                lineLable = []
                for la in line["labels"]:
                    lineLable.append(int(la))
                labels.append(lineLable)
                review.append(line["keyOpcode"])

            reviews = [line.strip().split() for line in review]
            print("len(labels): ", len(labels))
            print("len(reviews): ", len(reviews))

            return reviews, labels

    def _wordToIndex(self, reviews, word2idx):
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate):
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")
        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews):
        allWords = [word for review in reviews for word in review]
        subWords = [word for word in allWords if word not in self.stopWordDict]
        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        words = [item[0] for item in sortWordCount if item[1] >= 5]
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        word2idx = dict(zip(vocab, list(range(len(vocab)))))
        with open("../sc_data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        return word2idx

    def _getWordEmbedding(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/smcontractt/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        for word in words:
            try:
                vector = wordVec[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        self._readStopWord(self._stopWordSource)
        reviews, labels = self._readData(self._dataSource)
        word2idx = self._genVocabulary(reviews)
        word2idx = self._genVocabulary(reviews)
        reviewIds = self._wordToIndex(reviews, word2idx)
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labels, word2idx,
                                                                                    self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

data = Dataset(config)
data.dataGen()

def nextBatch(x, y, batchSize):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    numBatches = len(x) // batchSize
    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
        yield batchX, batchY

# 训练模型
# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels
wordEmbedding = data.wordEmbedding
labelList = data.labelList

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
    sess = tf.Session(config=session_conf)
    # 定义会话
    with sess.as_default():
        lstm = BiLSTMAttention(config, wordEmbedding)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        gradsAndVars = optimizer.compute_gradients(lstm.loss)
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))
        logFile.write("Writing to {}\n".format(outDir) + "\n")

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        # 保存模型的一种方式，保存为pb文件
        savedModelPath = "../model/bilstm-atten/savedModel"
        if os.path.exists(savedModelPath):
            shutil.rmtree(savedModelPath)
            os.mkdir(savedModelPath)
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, step, loss, predictions = sess.run([trainOp, globalStep, lstm.loss, lstm.predictions], feed_dict)
            timeStr = datetime.datetime.now().isoformat()

            acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions[:, 0], true_y=batchY[:, 0])
            accM, precisionM, recallM, f1M = multiPartEvl(batchY[:,1:], predictions[:,1:])

            return loss, acc, prec, recall, f_beta, accM, precisionM, recallM, f1M

        def devStep(batchX, batchY):
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: 1.0
            }
            step, loss, predictions = sess.run(
                [globalStep, lstm.loss, lstm.predictions],
                feed_dict)

            acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions[:, 0], true_y=batchY[:, 0])
            accM, precisionM, recallM, f1M = multiPartEvl(batchY[:, 1:], predictions[:, 1:])

            return loss, acc, precision, recall, f_beta, accM, precisionM, recallM, f1M

        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            logFile.write("start training model\n")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta, accM, precisionM, recallM, f1M = trainStep(batchTrain[0], batchTrain[1])
                currentStep = tf.train.global_step(sess, globalStep)
                print("trainOne: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                print("trainMulti: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, accM, recallM, precisionM, f1M))
                logFile.write("trainOne: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta) + "\n")
                logFile.write("trainMulti: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, accM, recallM, precisionM, f1M))

                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    logFile.write("\nEvaluation:\n")
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []
                    accsM = []
                    f_betasM = []
                    precisionsM = []
                    recallsM = []
                    i = 0
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        i += 1
                        print(i)
                        loss, acc, precision, recall, f_beta, accM, precisionM, recallM, f1M = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)
                        accsM.append(accM)
                        f_betasM.append(f1M)
                        precisionsM.append(precisionM)
                        recallsM.append(recallM)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))

                    logFile.write("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)) + "\n")
                    print("{}, step: {}, loss: {}, accM: {},precisionM: {}, recallM: {}, f_betaM: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accsM),
                                                                                                         mean(
                                                                                                             precisionsM),
                                                                                                         mean(recallsM),
                                                                                                         mean(f_betasM)))

                    logFile.write(
                        "{}, step: {}, loss: {}, accM: {},precisionM: {}, recallM: {}, f_betaM: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accsM),
                                                                                                       mean(precisionsM),
                                                                                                       mean(recallsM),
                                                                                                       mean(
                                                                                                           f_betasM)) + "\n")

                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/Bi-LSTM-atten/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
                    logFile.write("Saved model checkpoint to {}\n".format(path) + "\n")


        # 下面用于模型的保存和载入
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lstm.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()


sequenceLength = config.sequenceLength
testDataPath = config.testDataSource
word2idxPath = "../sc_data/wordJson/word2idx.json"
modelPath = "../model/Bi-LSTM-atten/model/"

from test import testCode
res = testCode(sequenceLength, testDataPath, word2idxPath, modelPath)
print("多标签的指标: ", res)
logFile.write("多标签的指标" + res + "\n")





