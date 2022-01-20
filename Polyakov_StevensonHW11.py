import numpy as np
import tensorflow as tf
import json
import timeit
from matplotlib import pyplot as plt
from playsound import playsound
hParams = {
    'experimentName': None,
    'datasetProportion': 1,
    'valProportion': .1,
    'numEpochs': 20,
    'denseLayers': None,
    "optimizer" : None,
    "convLayers" : []
    }
extraV1 = {
    "itemsToPlot" : [
        'C[32,64] d0.0 D[128,10] rms',
        'C[32,64] d0.2 D[128,10] rms',
        'C[32,64] d0.0 D[128,10] adam',
        'C[32,64] d0.2 D[128,10] adam',
    ]
}
extraV2 = {
    "itemsToPlot" : [
        'C[32,64] d0.0 D[128,10] adam',
        'C[32,64] d0.02 D[128,10] adam',
        'C[32,64] d0.05 D[128,10] adam',
        'C[32,64] d0.1 D[128,10] adam',
        'C[32,64] d0.2 D[128,10] adam',
        'C[32,64] d0.3 D[128,10] adam',
    ]
}
extraV3 = {
    "itemsToPlot" : [
        'C[32] d0.2 D[128,10] adam',
        'C[32,64] d0.2 D[128,10] adam',
        'C[32,64,128] d0.2 D[128,10] adam',
    ]
}
extraV4 = {
    "itemsToPlot" : [
        'C[32,64] d0.2 D[128,10] adam',
        'C[32,64] d0.2 D[256,128,10] adam',
        'C[32,64] d0.2 D[512,256,128,10] adam',
    ]
}
Counter = { "X" : 1}
def runExp(expNames = []):
    x_train, y_train, x_val, y_val, x_test, y_test = get10ClassData(flatten=False)
    for currExp in expNames:
        getHParams(currExp)
        trainRes, testRes = cnnGray([x_train, y_train, x_val, y_val, x_test, y_test])
        writeExperimentalResults(trainRes,testRes)
def getHParams(expName=None):
    shortTest = False # hardcode to True to run a quick debugging test
    if shortTest:
        print("+++++++++++++++++ WARNING: SHORT TEST +++++++++++++++++")
        hParams['datasetProportion'] = 0.01
        hParams['numEpochs'] = 2
    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams
    hParams["experimentName"] = expName
    if (expName == 'C[32,64] d0.0 D[128,10] rms'):
        dropProp = 0.0
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'
    elif (expName == 'C[32,64] d0.2 D[128,10] rms'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'
    elif (expName == 'C[32,64] d0.0 D[128,10] adam'):
        dropProp = 0.0
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64] d0.2 D[128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C[32,64] d0.02 D[128,10] adam'):
        dropProp = 0.02
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64] d0.05 D[128,10] adam'):
        dropProp = 0.05
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64] d0.1 D[128,10] adam'):
        dropProp = 0.1
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64] d0.3 D[128,10] adam'):
        dropProp = 0.3
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C[32] d0.2 D[128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64,128] d0.2 D[128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        {'conv_numFilters': 128, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2, 'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64,128,256] d0.2 D[128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,'drop_prop': dropProp},
        {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,'drop_prop': dropProp},
        {'conv_numFilters': 128, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,'drop_prop': dropProp},
        {'conv_numFilters': 256, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C[32,64] d0.2 D[256,128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
            {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,
             'drop_prop': dropProp},
            {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,
             'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [256, 128, 10]
        hParams['optimizer'] = 'adam'
    elif (expName == 'C[32,64] d0.2 D[512,256,128,10] adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
            {'conv_numFilters': 32, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,
             'drop_prop': dropProp},
            {'conv_numFilters': 64, 'conv_f': 3, 'conv_p': 'same', 'conv_act': 'relu', 'pool_f': 2, 'pool_s': 2,
             'drop_prop': dropProp},
        ]
        hParams['denseLayers'] = [512,256,128,10]
        hParams['optimizer'] = 'adam'
    else:
        #error…
        pass
def buildValAccuracyPlot(extraV):
    itemsToPlot = extraV
    yList = []
    for item in itemsToPlot:
        print(item)
        hParamsl, trainResults, testResults = readExperimentalResults(item)
        yList.append(trainResults["val_accuracy"])
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
               yList=yList,
               xLabel="Epoch",
               yLabelList=itemsToPlot,
               title="validation accuracies")
def buildTestAccuracyPlot(extraV):
    pointLabels = extraV
    xList = []
    yList = []
    xLabel = "Parameter Count"
    yLabel = "Test Set Accuracy"
    title = "Test Accuracy for Various Models"
    filename = "test accuracies for " + str(pointLabels)
    for item in pointLabels:
        hParamsl, trainResults, testResults = readExperimentalResults(item)
        xList.append(hParamsl["paramCount"])
        yList.append((testResults[1]))
    plotPoints(xList, yList, pointLabels, xLabel, yLabel, title, filename)
#given function
def processResults():
    hParamsl, trainResults, testResults = readExperimentalResults(hParams["experimentName"])
    itemsToPlot = ['accuracy', 'val_accuracy']
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
    yList=[trainResults[item] for item in itemsToPlot],
    xLabel="Epoch",
    yLabelList=itemsToPlot,
    title=hParamsl['experimentName'])
    itemsToPlot = ['loss', 'val_loss']
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
    yList=[trainResults[item] for item in itemsToPlot],
    xLabel="Epoch",
    yLabelList=itemsToPlot,
    title=hParamsl['experimentName'])
#provided function
def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + str(Counter["X"]) + "/" + "test Accuracies" + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)
#provided function : plots our data
def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    filepath = "results/" +str(Counter["X"]) + "/" +  str(title) + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)
#write expirement result
def writeExperimentalResults(trainResults,testResults):
    dict = {
        "hParams" : hParams,
        "trainResults" :trainResults,
        "testResults" : testResults
    }
    with open ("results/" + str(Counter["X"]) + "/" + str(hParams["experimentName"])+ ".txt" , "w") as resultFile:
        resultFile.write(json.dumps(dict))
#read data from our files
def readExperimentalResults(filePath):#Finish
    with open("results/" +  str(Counter["X"]) + "/" +  filePath + ".txt") as file:
        data = json.load(file)
        hParamsl = data["hParams"]
        return hParamsl, data["trainResults"], data["testResults"]
#shuffle data function
def correspondingShuffle(x,y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    x = tf.gather(x, shuffled_indices)
    y= tf.gather(y, shuffled_indices)
    return x,y
#get Data functions
def get10ClassData(proportion = 1.0,silent=True,flatten=True):
    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.fashion_mnist.load_data()
    if proportion != 1.0:
        x_train = x_train[:int(len(x_train) * proportion)]
        y_train = y_train[:int(len(y_train) * proportion)]
        x_test = x_test[:int(len(x_test) * proportion)]
        y_test = y_test[:int(len(y_test) * proportion)]
    #set rnage to be 0 - 1
    x_train = x_train / 255
    x_test = x_test / 255
    #reshape from 3d to 1d
    if flatten:
        x_train = tf.reshape(x_train,[x_train.shape[0],28*28])
        x_test = tf.reshape(x_test,[x_test.shape[0],28*28])
    #shuffle
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)
    #split data into train, val and test:
    if hParams["valProportion"] > 0.0:
        maxValIndexX = int(x_train.shape[0] * hParams["valProportion"])
        maxValIndexY = int(y_train.shape[0] * hParams["valProportion"])
        x_val = x_train[:maxValIndexX]
        y_val = y_train[:maxValIndexY]
        x_train = x_train[maxValIndexX:]
        y_train = y_train[maxValIndexY:]
        if not silent:
            print(x_train)
            print("With shape : " + str(x_train.shape))
            print(y_train)
            print("With shape : " + str(y_train.shape))
            print(x_test)
            print("With shape : " + str(x_test.shape))
            print(y_test)
            print("With shape : " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test
    if not silent:
        print(x_train)
        print("With shape : " + str(x_train.shape))
        print(y_train)
        print("With shape : " + str(y_train.shape))
        print(x_test)
        print("With shape : " + str(x_test.shape))
        print(y_test)
        print("With shape : " + str(y_test.shape))
    return x_train, y_train,None, None, x_test, y_test
#Model that works with grey scale images
def cnnGrayHardcode(dataSubsets):
    x_train, y_train, x_val, y_val, x_test, y_test = dataSubsets
    num_channels = 1
    image_width = 28
    image_height = 28
    x_train = tf.reshape(x_train, (-1, image_width, image_height, num_channels))
    x_val = tf.reshape(x_val, (-1, image_width, image_height, num_channels))
    x_test = tf.reshape(x_test, (-1, image_width, image_height, num_channels))

    fourthModel = tf.keras.Sequential()

    fourthModel.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(image_width, image_height, num_channels)
    ))
    fourthModel.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ))
    fourthModel.add(tf.keras.layers.Dropout(.2))

    fourthModel.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ))
    fourthModel.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ))
    fourthModel.add(tf.keras.layers.Dropout(.2))

    fourthModel.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ))
    fourthModel.add(tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ))
    fourthModel.add(tf.keras.layers.Dropout(.2))

    fourthModel.add(tf.keras.layers.Flatten())

    fourthModel.add(tf.keras.layers.Dense(128,activation="relu"))
    fourthModel.add(tf.keras.layers.Dense(10))

    startTime = timeit.default_timer()
    fourthModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics="accuracy",
                         optimizer=hParams["optimizer"])
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Construction Time : " + str(elapsedTime) + "\033[0;0m")

    startTime = timeit.default_timer()
    hist = fourthModel.fit(x_train, y_train,
                            validation_data=(x_val, y_val) if hParams['valProportion'] != 0.0 else None,
                            epochs=hParams['numEpochs'], verbose=1)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Training time : " + str(elapsedTime) + "\033[0;0m")
    hParams["paramCount"] = fourthModel.count_params()

    startTime = timeit.default_timer()
    accuracy = fourthModel.evaluate(x_test, y_test)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Testing time : " + str(elapsedTime) + " With Accuracy : " + str(accuracy) + "\033[0;0m")
    print('\033[92m')
    fourthModel.summary()
    print(".count_params() = " + str(fourthModel.count_params()))
    print("\033[0;0m")
    return hist.history, accuracy
# model for gray scale images that can be configured
def cnnGray(dataSubsets):
    x_train, y_train, x_val, y_val, x_test, y_test = dataSubsets
    num_channels = 1
    image_width = 28
    image_height = 28
    x_train = tf.reshape(x_train, (-1, image_width, image_height, num_channels))
    x_val = tf.reshape(x_val, (-1, image_width, image_height, num_channels))
    x_test = tf.reshape(x_test, (-1, image_width, image_height, num_channels))

    fourthModel = tf.keras.Sequential()
    for x in hParams["convLayers"]:
        x2 = x['conv_numFilters']
        fourthModel.add(tf.keras.layers.Conv2D(
            filters=x2,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(image_width, image_height, num_channels)
        ))
        fourthModel.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ))
        x3 = x["drop_prop"]
        fourthModel.add(tf.keras.layers.Dropout(x3))



    fourthModel.add(tf.keras.layers.Flatten())

    for x in range(len(hParams["denseLayers"]) - 1):

        fourthModel.add(tf.keras.layers.Dense(hParams["denseLayers"][x], activation="relu"))

    fourthModel.add(tf.keras.layers.Dense(10))

    startTime = timeit.default_timer()
    fourthModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics="accuracy",
                        optimizer=hParams["optimizer"])
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Construction Time : " + str(elapsedTime) + "\033[0;0m")

    startTime = timeit.default_timer()
    hist = fourthModel.fit(x_train, y_train,
                           validation_data=(x_val, y_val) if hParams['valProportion'] != 0.0 else None,
                           epochs=hParams['numEpochs'], verbose=1)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Training time : " + str(elapsedTime) + "\033[0;0m")
    hParams["paramCount"] = fourthModel.count_params()

    startTime = timeit.default_timer()
    accuracy = fourthModel.evaluate(x_test, y_test)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Testing time : " + str(elapsedTime) + " With Accuracy : " + str(accuracy) + "\033[0;0m")
    print('\033[92m')
    fourthModel.summary()
    print(".count_params() = " + str(fourthModel.count_params()))
    print("\033[0;0m")
    return hist.history, accuracy
def main():
    theSeed = 50
    np.random.seed(theSeed)
    tf.random.set_seed(theSeed)
    buildTestAccuracyPlot()
def main2():
    theSeed = 50
    np.random.seed(theSeed)
    tf.random.set_seed(theSeed)
    '''
    runExp(extraV1["itemsToPlot"])
    for x in extraV1["itemsToPlot"]:
        hParams["experimentName"] = x
        processResults()
    buildValAccuracyPlot(extraV1["itemsToPlot"])
    buildTestAccuracyPlot(extraV1["itemsToPlot"])
    '''
    Counter["X"] = Counter["X"] + 1
    '''
    runExp(extraV2["itemsToPlot"])
    for x in extraV2["itemsToPlot"]:
        hParams["experimentName"] = x
        processResults()
    buildValAccuracyPlot(extraV2["itemsToPlot"])
    buildTestAccuracyPlot(extraV2["itemsToPlot"])
    '''
    Counter["X"] = Counter["X"] + 1

    #runExp(extraV3["itemsToPlot"])
    for x in extraV3["itemsToPlot"]:
        hParams["experimentName"] = x
        processResults()
    buildValAccuracyPlot(extraV3["itemsToPlot"])
    buildTestAccuracyPlot(extraV3["itemsToPlot"])
    Counter["X"] = Counter["X"] + 1

    runExp(extraV4["itemsToPlot"])
    for x in extraV4["itemsToPlot"]:
        hParams["experimentName"] = x
        processResults()
    buildValAccuracyPlot(extraV4["itemsToPlot"])
    buildTestAccuracyPlot(extraV4["itemsToPlot"])
    playsound("./sound.mp3")
main2()