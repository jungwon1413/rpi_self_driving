# -*- coding: utf-8 -*-

### 이 코드는 Github의 Shmuelnaaman 리포지토리를 참고해서 만들었음을
### 미리 밝힘니다. - 2016. 12. (Jungwon Kim)
### 출처: https://github.com/Shmuelnaaman/Traffic_Sign_Classifier_neural_network


########################################
### Code First written on 2016.12.6  ###
### Code origined by Shmuelnaaman    ###
### Code re-designed by Jungwon Kim  ###
########################################



###################################
### STEP 1: Dataset Exploration ###
###################################

# Load Libraries
# import cv2		### Imported for video encoding and decoding
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pylab
from parameter import *



# Initializing the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())



train_acc_batch = []
val_acc_batch = []
train_cost_batch = []
batches = []
# Launch the graph

# sess.run(init)
    # Training cycle
for epoch in range(training_epochs):

	######################################
    for offset in range(0, train_labels.shape[0], batch_size):
    # Get a batch of training features and labels
        end = offset + batch_size
        batch_features = train_features[offset:end]
        batch_labels = train_labels[offset:end]

			##################################
			# Run optimizer and get loss
        sess.run([optimizer],feed_dict={x: batch_features,y: batch_labels})
    if epoch % display_step == 0:
        batches.append(offset)
        c = sess.run(cost, feed_dict={x: batch_features, y: batch_labels})
        print('Epoch {:>2}/{}'.format(epoch+1, training_epochs), "cost=", "{:.5f}".format(c))
        a = sess.run(  accuracy, feed_dict={x: batch_features, y: batch_labels})
# a = sess.run(accuracy(feed_dict={x:batch_features, y: batch_labels})  )
        print("Accuracy train:", "{:.5f}".format(a))
        train_cost_batch.append([c])
        train_acc_batch.append([a])

        a = sess.run(  accuracy, feed_dict = {x: valid_features, y: valid_labels})

        print("Accuracy Val:", "{:.5f}".format(a))


        val_acc_batch.append([a])

    # Test model
a = sess.run(  accuracy, feed_dict={x: X_test_t, y: y_test})
print("Accuracy:", a)


save_path = tf.train.Saver().save(sess, "./Dataset_Backup/model.ckpt")
print("Model saved in file: %s" % save_path)


"""
# 여기까지의 상황 저장
saver = tf.train.Saver()
save_path = saver.save(sess, './sign_recognition.ckpt')
"""
