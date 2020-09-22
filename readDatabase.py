import os
import gzip
import numpy as np
import matplotlib.pyplot  as plt

def readDatabase(path, kind='train'):
    
    """Load MNIST Fashion data from `path`"""

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz' 
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28,28)

    return images, labels



if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    [trainData, trainLabels] = readDatabase(path)
    [testData, testLabels] = readDatabase(path, 'test')

    """Test MNIST Fashion data """
    #labels map
    labelsMap = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    # plot some images to make sure that the labels and images 
    fig, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(trainData[0])
    axarr[0,0].set_title(labelsMap[trainLabels[0]])
    axarr[0,1].imshow(trainData[1])
    axarr[0,1].set_title(labelsMap[trainLabels[1]])
    axarr[1,0].imshow(testData[0])
    axarr[1,0].set_title(labelsMap[testLabels[0]])
    axarr[1,1].imshow(testData[1])
    axarr[1,1].set_title(labelsMap[testLabels[1]])
    plt.show()

