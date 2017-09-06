import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import parameter as prd
import socket



sess = tf.Session()
tf.train.Saver().restore(sess, "./Dataset_Backup/model.ckpt")
print("Model restored.")



### Visualization for the cost and accuracy (y-axis)
### with respect to the "training epochs"(x-axis).
#batches=    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 17, 18 ,19, 20, 21, 22, 23, 24 ]

def graph_cost_and_accuracy():
    batches = []
    insert = 0
    while insert < training_epochs:
        batches.append(insert)
        insert += 1;

    loss_plot = plt.subplot(211)
    loss_plot.set_title('cost')
    loss_plot.plot(batches, train_cost_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, val_acc_batch, 'b', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()

# 분석 비용 & 정확도 그래프 형식 준비
# graph_cost_and_accuracy()
# plt.show()



#요 위치쯤에 내 코드가 들어가면 된다 opencv


def jungwon():

	### Load the images and plot them here.
	### Feel free to use as many code cells as needed.

	### Run the predictions here.
	### Feel free to use as many code cells as needed.
	from scipy import misc
	import matplotlib.pyplot as plt
	import os
	# import pandas as pd		<< 나중에 그래프 정리때 쓰임.
	# %matplotlib inline

	imgs = []
	image_count = 0
	img_list = []

	for filename in os.listdir('signs'):
		img = plt.imread('signs/' + filename)
		imgs.append(img)
		img_list.append(image_count)
		image_count += 1	# 이미지의 갯수를 센다.
		print(filename, img.shape)



	# 이미지의 최종 갯수는 1개를 더한 수이다. (0부터 시작되므로)
	image_count += 1



	def input_images(img_list):
		fig = plt.figure()
		for fig_count in img_list:
			fig.add_subplot(2,3,fig_count+1)
			plt.imshow(imgs[fig_count])


	# 입력 이미지를 화면의 띄워주는 함수
	#	input_images(img_list)
	#	plt.show()


	# 아래는 input_images 함수의 원래 입력 형태이다.

	A = np.array(imgs)
	A_t = np.zeros((A.shape[0],A.shape[1]*A.shape[2],A.shape[3]),dtype=float)

	for i in range (A.shape[3]):
		A_t[:,:,i] = prd.normalize_scale(A[:,:,:,i].reshape(A.shape[0],
		A.shape[1] * A.shape[2]))

	A_t = A_t.reshape(A_t.shape[0], A_t.shape[1] * A_t.shape[2])


	preds = sess.run(prd.predict, feed_dict = {prd.x: A_t})
	print(preds)



	# X축에 해당되는 리스트 작성
	img_list_x = [element+1 for element in img_list]



	def image_prediction(A, label_type, preds, img_list_x):
		fig = plt.figure()
		for i in img_list_x:
			fig.add_subplot(2,image_count,i)
			plt.imshow(A[i-1,:,:,:])
			plt.axis('off')
			fig.add_subplot(2,image_count,i+image_count)
			plt.imshow(label_type[preds[i-1]])
			plt.axis('off')



	# 가장 가까운 것으로 예상되는 이미지를 보여주는 함수 (원본 + 예상)
	#	image_prediction(A, label_type, preds, img_list_x)
	#	plt.show()



	### Visualize the softmax probabilities here.
	### Feel free to use as many code cells as needed.
	top_k_preds = sess.run(tf.nn.top_k(prd.prediction, k=5), feed_dict={prd.x: A_t})
	print(top_k_preds.values)
	print(top_k_preds.indices)



	img_list_y = img_list		# Y축에 해당되는 숫자 리스트 작성



	def image_prediction_detailed(A, top_k_preds, image_count, 
							  img_list_x, img_list_y):
		fig = plt.figure()
		for i in img_list_x:
			fig.add_subplot(6,image_count,i)
			plt.imshow(A[i-1,:,:,:])
			plt.axis('off')
			for j in img_list_y:
				fig.add_subplot(6,image_count,i+(j+1)*image_count)
				plt.imshow(label_type[top_k_preds.indices[i-1][j]])
				plt.axis('off')

	# 상위 후보 5개 예상 목록을 보여주는 함수
	#	image_prediction_detailed(A, top_k_preds, image_count, img_list_x, img_list_y)
	#	plt.show()

	def prediction_certainty_graph(top_k_preds):
		import pandas as pd
		A = pd.DataFrame(top_k_preds.values.transpose())

		A.plot(kind='bar').set_ylabel('Probability')

	# 예측의 정확도를 추측하는 그래프를 보여주는 함수
	#	prediction_certainty_graph(top_k_preds)
	#	plt.show()

	for classify in top_k_preds.indices:
		if classify[0] <= 14:
			return '0'		# Stop
		elif classify[0] == 33:
			return '3'		# 오른쪽
		elif classify[0] == 34:
			return '1'		# 왼쪽
		elif classify[0] == 35:
			return '2'		# 직진
		elif classify[0] == 25:
			return '0'		# 공사중도 스탑값으로
		elif classify[0] == 40:
			return '5'		# 공사중도 스탑값으로
		else:
			return '255'		# 그 외의 값은 255(에러) 송출


def mask_hsv_red(img):

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_low_red = np.array([0,100,40])
	upper_low_red = np.array([2,255,255])
	mask_low_red = cv2.inRange(hsv, lower_low_red,upper_low_red)

	lower_up_red = np.array([165,100,40])
	upper_up_red = np.array([180,255,255])
	mask_up_red = cv2.inRange(hsv,lower_up_red,upper_up_red)

	mask_red = cv2.add(mask_low_red,mask_up_red)
	return mask_red

def find_traffic_sign(img):

	mask_red = mask_hsv_red(img)

	some, contours, hierarchy = cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		if cv2.contourArea(contour) > 600:
			epsilon = 0.03*cv2.arcLength(contour,True)
			approx = cv2.approxPolyDP(contour,epsilon,True)
			if len(approx) == 8:
				x,y,w,h = cv2.boundingRect(contour)
				sign = img[y:y+h, x:x+w]
				sign = cv2.resize(sign,(32,32))
				#sign = cv2.flip(sign,1)
				img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
				return sign.copy()

def save_sign(img):
	for i in range(1,6):
		cv2.imwrite('signs/sign_' + str(i) + '.jpg',img)
		#if j%3000 == 1:
			#print(j)
			#jungwon() 

"""
j = 0
def main():
	global j
	#cap = cv2.VideoCapture('http://192.168.1.8:8080/stream/video.mjpeg')
	cap = cv2.VideoCapture('sample.avi')
	while True:
		#print(j)
		ret, frame = cap.read()
		if not ret:
			print('aaa')
			break
		#frame = cv2.flip(frame,1)

		sign = find_traffic_sign(frame)

		cv2.imshow('frame', frame)
		if sign is not None:
			cv2.imshow('sign',sign)
			save_sign(sign)
			j+=1
			if j%25 == 24:
				#print(j)
				jungwon()
		

		if cv2.waitKey(30) & 0xff == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
while True:
	main()

"""


port = 60000
s = socket.socket()
host = '192.168.1.12'
s.bind((host,port))
s.listen(5)

#fnames = ["sign_1.jpg", "sign_2.jpg", "sign_3.jpg", "sign_4.jpg", "sign_5.jpg"]

print("Server listening....")

def server():
	while True:
		conn, addr = s.accept()
		with open('signs/sign_1.jpg', 'wb') as f:
			print("file opened")
			while True:
				data = conn.recv(2048)
				#print("data = %s", (data))
				if not data:
					#f.close()
					break
				f.write(data)
		conn.close()
		#jungwon()
		send_result(jungwon())
	s.close()

def send_result(data):
	conn, addr = s.accept()
	#data = "result"
	conn.send(data.encode("UTF-8"))
	conn.close()



def main():

	#classify_code = None
	server()
	#jungwon()
	#send_result()


while True:
	main()
