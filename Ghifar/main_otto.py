'''
	main_otto.py: the main file for training and testing Otto data
'''
from loader import *
from LogisticRegression import *
import cPickle as pickle


def train_lr(learning_rate=9e-2, l2reg=0.0, batch_size=500, n_epochs=5000, outfile='outputs.csv'):
	'''
	Training Otto data using Logistic Regression
 	Create by Ghifar
 	Tuesday, 14/04/2015
	'''
	print '[train_lr] learning_rate : ',learning_rate
	print '           l2reg : ',l2reg
	print '           batch_size : ',batch_size
	print '           n_epochs : ',n_epochs
	# Load training set (with contrast normalization preprocessing)
	# (x_train, y_train), mu, std = load_otto_data(filepath='train.csv')
	# (x_train, y_train), mu, std = load_otto_data(filepath='train.csv', preprocess='cn')
	(x_train, y_train), mu, std = load_otto_data(filepath='train.csv', preprocess='0-1')
	
	# # Load test set
	x_test=None
	y_test=None
	# (x_test, y_test), mu, std = load_otto_data(filepath='test.csv',preprocess='cn',mu=mu, std=std)
	(x_test, y_test), mu, std = load_otto_data(filepath='test.csv', preprocess='0-1')
	# print y_test.eval()

	# Contruct and compile LR model
	n_in = 93
	n_out = 9
	print x_train.eval().shape
	print y_train.eval().shape
	lr = LogisticRegression(n_in=n_in, n_out=n_out)

	lr.compile(
		train_set_x=x_train, train_set_y = y_train, #training set
        valid_set_x=x_train, valid_set_y = y_train, #validation set
        test_set_x=x_test, # test set
        learning_rate=learning_rate,
        batch_size=batch_size
    )

	# Training with SGD
	lr.training_sgd(n_epochs=n_epochs)

	# predict
	print 'Predict the label...'
	y_predict = lr.test_model(x_test.eval())

	print 'Save the prediciton: ',outfile
	write_otto_output(y_predict,filename=outfile)

	print '[train_lr] learning_rate : ',learning_rate
	print '           l2reg : ',l2reg
	print '           batch_size : ',batch_size
	print '           n_epochs : ',n_epochs



if __name__ == '__main__':
	train_lr(outfile='outputs_lr9e-2_ba500_ep5000_noprep.csv')