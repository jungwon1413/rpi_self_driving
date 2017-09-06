###################################
### STEP 1: Dataset Exploration ###
###################################

# Load Libraries
# import cv2		### Import for video encoding and decoding
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pylab

#%matplotlib inline	<< jupyter command
#%pylab inline
pylab.rcParams['figure.figsize'] = (14, 7)


# Load pickled data

# TODO: fill this in based on where you saved the training and testing data
training_file = './lab_2_data/train.p'
testing_file = './lab_2_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


### To start off, let's do a basic data summary

# TODO: number of training examples
n_train = len (X_train)

# TODO: number of testing examples
n_test = len (X_test)

# TODO: what's the shape of an image?
image_shape = X_train.shape[1:]

# TODO: how many classes are in the dataset
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape=", image_shape)
print("Number of classes =", n_classes)


# 위 코드의 예상 출력 결과
"""
Result::
    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
"""

##########################################################################
### Visualization that demonstrates the available types of traffic signs.#
##########################################################################

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.

label_type = []
# 입력 데이터의 유형을 보여준다.
for i in range(0, n_classes):
    plt.subplot(7, 7, i+1)
    label_type.append(X_test[np.where( y_test == i )[0][0],:,:,:].squeeze())
    img = plt.imshow(X_test[np.where( y_test == i )[0][0],:,:,:].squeeze())
    plt.axis('off')
### 함수로 만들고 싶을때 입력인자로 필요한 요소들
### n_classes, label_type, X_test, y_test
### img와 label_type가 리턴이 되어야한다.

# plt.show()		# << 각 숫자별 할당된 이미지 샘플 보여주기

################################################################
### Visualization that demonstrates the occurence distribution #
### for different types of traffic signs.		               #
################################################################

def input_data_graph():
	# Two subplots, the axes array is l-d
	f, axarr = plt.subplots(2, sharex=True)
	axarr[0].hist(y_train, bins=n_classes)
	axarr[0].set_title('Label Train Distribution')
	axarr[0].set_xlabel('Labels')
	axarr[0].set_ylabel('Count')
	axarr[1].hist(y_test,bins=n_classes)
	axarr[1].set_title('Label Test Distribution')
	axarr[1].set_xlabel('Labels')
	axarr[1].set_ylabel('Count')
	plt.show()

# 데이터 입력의 클래스별 분포를 보여준다.
# input_data_graph()		<< 실제 구동에는 불필요하다.

##################################################
### STEP 2: Design and Test a Model Architecture #
##################################################


### Preprocess the data here.
### Feel free to use as many code cells as needed.


### Normalize the figures ###

# Implement Min-Max scaling for image data
def normalize_scale(image_data):
    """
	Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
	:param image data: The image data to be normalized
	:return: Normalized image data
    """
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255
    return (a + ( ( (image_data - greyscale_min)*(b - a) )/(greyscale_max - greyscale_min) ) )



X_test_t = np.zeros((X_test.shape[0], X_test.shape[1] * X_test.shape[2], X_test.shape[3]), dtype=float)
X_train_t = np.zeros((X_train.shape[0], X_train.shape[1] * X_train.shape[2], X_train.shape[3]), dtype=float)



for i in range (X_test.shape[3]):
    X_test_t[:,:,i] = normalize_scale(X_test[:,:,:,i].reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    X_train_t[:,:,i]= normalize_scale(X_train[:,:,:,i].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

X_test_t = X_test_t.reshape(X_test_t.shape[0], X_test_t.shape[1] * X_test_t.shape[2])
X_train_t = X_train_t.reshape(X_train_t.shape[0], X_train_t.shape[1] * X_train_t.shape[2])


### Transform labels using One-Hot Encoding ###
from sklearn.preprocessing import LabelBinarizer

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

# Change to float32, so it can be multiplied against the features
# in Tensorflow, which are float32
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
is_labels_encod = True

print('Labels One-Hot Encoded')

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

#########################################################################
### Splitting and randomize datasets for training and validation sets ###
#########################################################################
from sklearn.model_selection import train_test_split

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
	    X_train_t,
	    y_train,
	    test_size=0.3,
	    random_state=0)


### Train your model here.
### Feel free to use as many code cells as needed.


######################################################
### Deep Neural Network in TensorFlow 2 hidden_layer #
######################################################

#### # Parameters

n_input = train_features.shape[1]
n_classes = train_labels.shape[1]

# Parameters
learning_rate = 0.0006
training_epochs = 45
batch_size = 1000
display_step = 1
#del logits

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


n_hidden_layer1 = 256	# 1st layer number of features (0 ~ 255)
n_hidden_layer2 = 256	# 2nd layer number of features (0 ~ 255)


weight_layer1 = tf.Variable(tf.random_normal([n_input,n_hidden_layer1], mean=0, stddev=0.01))
weight_layer2 = tf.Variable(tf.random_normal([n_hidden_layer1, n_hidden_layer2], mean=0, stddev=0.01))
weight_out = tf.Variable(tf.random_normal([n_hidden_layer2, n_classes]))


biases_layer1 = tf.Variable(tf.random_normal([n_hidden_layer1], mean=0, stddev=0.01))
biases_layer2 = tf.Variable(tf.random_normal([n_hidden_layer2], mean=0, stddev=0.01))
biases_out = tf.Variable(tf.random_normal([n_classes]))


def weight_parameter():
# Store layers weight & bias
	weights = {
	'hidden_layer1': weight_layer1,
	'hidden_layer2': weight_layer2,
	'out': weight_out}
	return weights

def biases_parameter():
	biases = {
	'hidden_layer1': biases_layer1,
	'hidden_layer2': biases_layer2,
	'out': biases_out}
	return biases


def hidden_layer1():
# Hidden layer 1 with ReLU activation
	layer_1 = tf.add(tf.matmul(x, weights['hidden_layer1']), biases['hidden_layer1'])
	layer_1 = tf.nn.relu(layer_1)
	return layer_1


def hidden_layer2(layer_1):
# Hidden layer 2 with ReLu activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['hidden_layer2'])
	layer_2 = tf.nn.relu(layer_2)
	return layer_2


def output_layer(layer_2):
# Output layer with linear activation
	logits = tf.matmul(layer_2, weights['out']) + biases['out']
	return logits


weights = weight_parameter()
biases = biases_parameter()

layer_1 = hidden_layer1()
layer_2 = hidden_layer2(layer_1)
logits = output_layer(layer_2)


prediction = tf.nn.softmax(logits)
predict = tf.argmax(logits, 1)






# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
