

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


##################### 改這裡 ####################

epochs = 10
batch_size = 8
margin = 1  # Margin for constrastive loss.
IMAGE_SIZE = 300
import cv2
import os
# OK樣本路徑
D = "D:\\harden\\dataset\\H1_water_de\\3ok\\"  
# NG樣本路徑
E = "D:\\harden\\dataset\\H1_water_de\\3ng\\"
##################### 改這裡 ####################
images1 = []
labels1 = []
dir_counts = 0
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    #get size
    h, w , _= image.shape
    
    #adj(w,h)
    longest_edge = max(h, w)    
    
    #size = n*n 
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    BLACK = [0, 0, 0]   
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    return cv2.resize(constant, (height, width))
def d (D=D,images=images1,labels=labels1):
    vou=0
    for i in os.listdir(D):
        
        try:
            #print(D+i)
            img1 = cv2.imread(D+i)
            #print(D+i)
            #img1 = cv2.resize(img1,(IMAGE_SIZE,IMAGE_SIZE))
            img1 = resize_image(img1, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img1)
            #print(0)
            labels.append(dir_counts+1) #  ,OK sample to label: 1
        except:
            print("error")
        vou +=1
        if vou >=20:
            break
    print("A already read")
    return(images,labels)
d(D,images1,labels1) # OK sample to label: 1
test_00b = images1[0]
def e (E=E,images=images1,labels=labels1):
    BC = 0
    for i in os.listdir(E):
        try:
            img2 = cv2.imread(E+i)
            #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img2)
            labels.append(dir_counts)
        except:
            print("error")
        BC = BC+1
        if BC == 120:
            break
    print("B already read")
    return(images,labels)
e(E,images1,labels1) # NG sample to label: 0

########## 有效測試集 #########
def e2 (E=E,images=images1,labels=labels1):
    BC = 0
    for i in os.listdir(E):
        try:
            img2 = cv2.imread(E+i)
            #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img2)
            labels.append(dir_counts)
        except:
            print("error")
        BC = BC+1
        if BC == 120:
            break
    print("B already read")
    return(images,labels)
def e3 (E=E,images=images1,labels=labels1):
    BC = 0
    for i in os.listdir(E):
        try:
            img2 = cv2.imread(E+i)
            #img2 = cv2.resize(img2,(IMAGE_SIZE,IMAGE_SIZE))
            img2 = resize_image(img2, IMAGE_SIZE, IMAGE_SIZE)
            images.append(img2)
            labels.append(dir_counts+1)
        except:
            print("error")
        BC = BC+1
        if BC == 120:
            break
    print("B already read")
    return(images,labels)
image_test1=[]
image_test2=[]
label_test1=[]
label_test2=[]
e2(E,image_test1,label_test1)
e2(D,image_test2,label_test2)
e3(D,image_test1,label_test1)

print("LAB",labels1)
from sklearn.model_selection import train_test_split
label = np.array(labels1)
X_train,X_test,y_train,y_test =  train_test_split(images1, label,test_size=0.2,random_state=42 )#
x_train = np.array(X_train, dtype=np.float32)
x_test = np.array(X_test, dtype=np.float32)
x_train = x_train/255.0
x_test  =  x_test/255.0

image_test1=np.array(image_test1, dtype=np.float32) /255.0
label_test1=np.array(label_test1)
print("label_test1",label_test1.shape)





def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.
    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.
    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

def make_pairs2(x, y):
    """Creates a tuple containing image pairs with corresponding label.
    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.
    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make validation pairs
pairs_val, labels_val = make_pairs(x_test, y_test)

# make test pairs
pairs_test, labels_test = make_pairs(image_test1, label_test1)



x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
x_train_2 = pairs_train[:, 1]

"""
Split the validation pairs
"""

x_test_1 = pairs_val[:, 0]  # x_test_1.shape = (60000, 28, 28)
x_test_2 = pairs_val[:, 1]

"""
Split the test pairs
"""

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
x_test_2 = pairs_test[:, 1]


"""
## Visualize pairs and their labels
"""


def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.
    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).
    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()



visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)


visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)



visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)



# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input = layers.Input((300, 300, 3))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(4, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(10, activation="tanh")(x)
embedding_network = keras.Model(input, x)


input_1 = layers.Input((300, 300, 3))
input_2 = layers.Input((300, 300, 3))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)




def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).
    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


"""
## Compile the model with the contrastive loss
"""

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()


"""
## Train the model
"""

history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_test_1, x_test_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)

"""
## Visualize results
"""


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.
    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.
    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

"""
## Evaluate the model
"""

results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

"""
## Visualize the predictions
"""

predictions = siamese.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)

"""
**Example available on HuggingFace**
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Siamese%20Network-black.svg)](https://huggingface.co/keras-io/siamese-contrastive) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Siamese%20Network-black.svg)](https://huggingface.co/spaces/keras-io/siamese-contrastive) |
"""