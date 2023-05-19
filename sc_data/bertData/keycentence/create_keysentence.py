import json


# 从智能合约操作码中获取关于所选取的五种漏洞的关键序列
# keyOpWords = {"ADD", "MUL", "SUB", "ADDMOD", "MULMOD", "EXP", "DELEGATECALL", "CALLVALUE", "CALL", "SELFDESTRUCT", "CALLDATALOAD",
#               "TIMESTAMP"}

# keyOpWords = {"ADD", "MUL", "SUB", "DELEGATECALL", "CALLVALUE", "CALL", "TIMESTAMP"}
# stopOpWords = {"STOP", "SELFDESTRUCT", "RETURN", "INVALID ", "SUICIDE", "JUMP", "JUMPI", "REVERT"}



def CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName):
    with open(oriFilePathandName, "r") as file:
        with open(createKeyPathandName, "w") as newFile:

            file_lines = json.load(file)
            resList = []
            beforeLen = []
            afterLen = []

            for line in file_lines:
                lineKeyCode = ""
                sc_opcode = str(line["contractCode"]).strip().split(" ")
                i = 0
                while i < len(sc_opcode):
                    if sc_opcode[i] in keyOpWords:
                        starIndex = i
                        while starIndex > 0:
                            if sc_opcode[starIndex] in stopOpWords:
                                break
                            starIndex -= 1
                        while i < len(sc_opcode)-1:
                            if sc_opcode[i] in stopOpWords:
                                break
                            i += 1
                        while starIndex <= i:
                            lineKeyCode += sc_opcode[starIndex] + " "
                            starIndex += 1
                    i += 1

                if len(lineKeyCode.strip().split(" ")) == 0:
                    a = 0
                    while a < 512:
                        lineKeyCode += sc_opcode[a] + " "
                        a += 1
                lineRes = {"lable": line["lable"], "contractCode": lineKeyCode.strip()}
                resList.append(lineRes)
                print("before", len(sc_opcode))
                print("after", len(lineKeyCode.strip().split(" ")), len(sc_opcode))
                beforeLen.append(len(sc_opcode))
                afterLen.append(len(lineKeyCode.strip().split(" ")))

            json.dump(resList,newFile)

            import matplotlib.pyplot as plt
            #     作图
            x_l = []
            for i in range(len(beforeLen)):
                x_l.append(i + 1)

            plt.bar(x_l, beforeLen, color='r')
            plt.bar(x_l, afterLen, color='b')
            plt.savefig("beforAfter.png")
            plt.show()

            file.close()
            newFile.close()

if __name__ == '__main__':
    keyOpWords = {"ADD", "MUL", "SUB", "ADDMOD", "MULMOD", "EXP"}
    stopOpWords = {"STOP", "SELFDESTRUCT", "RETURN", "INVALID ", "SUICIDE", "JUMP", "JUMPI", "JUMPDEST", "REVERT"}

    #
    # oriFilePathandName = "../opcode/Delegatecall_opcode_157.json"
    # createKeyPathandName = "./1/delegatecall.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)
    #
    # oriFilePathandName = "../opcode/good_sc_10000_opcode.json"
    # createKeyPathandName = "./2/good.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)
    #
    # oriFilePathandName = "../opcode/Integer_overflow_opcode_18263.json"
    # createKeyPathandName = "./3/integer.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)
    #
    # oriFilePathandName = "../opcode/Reentrancy_opcode_1422.json"
    # createKeyPathandName = "./4/ree.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)
    #
    # oriFilePathandName = "../opcode/Timestamp_opcode_777.json"
    # createKeyPathandName = "./5/timestamp.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)
    #
    # oriFilePathandName = "../opcode/Unchecked_call_opcode_666.json"
    # createKeyPathandName = "./6/unchackcall.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)

    # oriFilePathandName = "../opcode/good_sc_18000_opcode.json"
    # createKeyPathandName = "./2/good_18000.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)

    # oriFilePathandName = r"E:\graduate_design\textClassifier-master\sc_data\train_test_data\test.json"
    # createKeyPathandName = "./testKey.json"
    # CreateKeySence(oriFilePathandName, keyOpWords, stopOpWords, createKeyPathandName)










