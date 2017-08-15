
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import numpy as np

def plot_digit(digit):
	digit_img = digit.reshape(28,28)
	plt.imshow(digit_img,cmap=matplotlib.cm.binary,
		interpolation='nearest')
	plt.axis('off')


def main():

	mnist = fetch_mldata('MNIST original')
	X,Y = mnist["data"],mnist["target"]
	X_train,X_test,Y_train,Y_test=X[:60000],X[60000:],Y[:60000],Y[60000:]
	shuffle_index= np.random.permutation(60000)
	X_train,Y_train = X_train[shuffle_index],Y_train[shuffle_index]
	
	#Binary Classifier
	y_train_5 = (Y_train==5)
	y_test_5 = (Y_test == 5)

	sgdClf = SGDClassifier()
	sgdClf.fit(X_train,y_train_5)
	#This is for checking later on. I know that X[36015] is letter 5 by the image.
	number_5 = X[36015]
	plot_digit(number_5)
	plt.show()
	success = 0
	for trials in range(0,500):
		if sgdClf.predict([number_5])==True:
			success+=1 
	print("Probability: %f" %(success/500))

if __name__ == "__main__":
	main()
