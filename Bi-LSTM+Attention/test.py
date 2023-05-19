import json
import tensorflow as tf
from multiEvl import multiPartEvl
import numpy as np

def testCode(sequenceLength, testDataPath, word2idxPath, modelPath):
    with open(testDataPath, "r") as testFile:
        test_lines = json.load(testFile)
        pred_y = []
        true_y = []
        print("len(test_lines)", len(test_lines))
        i = 0
        for line in test_lines:
            if i < 7600:
                i += 1
                # 划分测试集
                continue
            x = str(line["keyOpcode"]).strip()
            with open(word2idxPath, "r", encoding="utf-8") as f:
                word2idx = json.load(f)
            xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
            if len(xIds) >= sequenceLength:
                xIds = xIds[:sequenceLength]
            else:
                xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))
            graph = tf.Graph()
            with graph.as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
                session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    checkpoint_file = tf.train.latest_checkpoint(modelPath)
                    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                    saver.restore(sess, checkpoint_file)
                    inputX = graph.get_operation_by_name("inputX").outputs[0]
                    dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]
                    predictions = graph.get_tensor_by_name("output/predictions:0")
                    pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
                    multiPre = []
                    for pred_lables in pred[1:]:
                        multiPre.append(int(pred_lables))
                    print("multiPre", multiPre)
                    pred_y.append(multiPre)
                    mulitLables = []
                    for lab in line["labels"][1:]:
                        mulitLables.append(int(lab))
                    true_y.append(mulitLables)
                    print("mulitLables", mulitLables)
        acc, precision, recall, f1 = multiPartEvl(np.array(true_y), np.array(pred_y))
        # print("testMulti: acc: {}, recall: {}, precision: {}, f_beta: {}".format(acc, recall, precision, f1))
        testFile.close()
        return "testMulti: acc: {}, recall: {}, precision: {}, f_beta: {}".format(acc, recall, precision, f1)
