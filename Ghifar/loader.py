'''
	Author :  Muhammad Ghifary
	Created at : 14/04/2015
	Description: 
		- Loader for the Otto dataset
		- Return 'numpy' or 'theano' variable type
		- Include some preprocessing step

To do :
		[] split the data into training and validation set
'''
import csv
import numpy as np
import theano
import theano.tensor as T


def load_otto_data(filepath='',datatype='theano',
					preprocess=None,mu=None,std=None):
	'''
	Load CSV data
		- filepath : 'train.csv' or 'test.csv', assume the file is in the same dir
		- datatype : 'theano' (return theano type) or 'numpy' (return numpy type)
		- preprocess: type of preprocessing that want to do
			- mu : None 
			- std : None
	'''
	if filepath == '':
		filepath = 'train.csv'
	
	f = open(filepath)
	csv_f = csv.reader(f)

	classes = ["Class_1","Class_2","Class_3","Class_4","Class_5",
				"Class_6","Class_7","Class_8","Class_9"]
	

	x_data_list = []
	y_data_list = []
	rownum = 0
	for row in csv_f:
		if rownum > 0:
			x_data_list.append(row[1:94])
			if filepath != 'test.csv':
				cls = classes.index(row[-1])
				y_data_list.append(classes.index(row[-1]))


		rownum += 1
	f.close()


	x_data = np.array(x_data_list).astype('float32')
	n = x_data.shape[0]

	y_data = np.zeros(n)
	if filepath != 'test.csv':
		y_data = np.array(y_data_list).astype('float32')


	# mu = None
	# std  = None

	print '-- before'
	print 'min: ',np.min(x_data)
	print 'max: ', np.max(x_data)

	if preprocess is not None:
		if preprocess == 'cn': #contrast normalization, i.e., to zero mean and unit variance
			print 'zero-mean and unit variance'
			x_data, mu, std = contrast_normalize(x_data, mu, std)
			print np.mean(x_data,axis=0)
			print np.std(x_data,axis=0)
		elif preprocess == '0-1':
			print '[0-1] normalization'
			x_data = zeroone_normalize(x_data)


		#elif ...
		#elif ...
		# ... <please add other kinds of preprocessing step here>
	print '-- after'
	print 'min: ',np.min(x_data)
	print 'max: ', np.max(x_data)	
	data_set = (x_data, y_data)
	if datatype == 'theano':
		return shared_dataset(data_set), mu, std
	else: # nunmpy
		return data_set, mu, std

def write_otto_output(y,filename='outputs.csv'):
	y_list = y.tolist()
	f = open(filename,'wt')

	try:
		writer = csv.writer(f)
		writer.writerow( ('id','Class_1','Class_2','Class_3',
						'Class_4','Class_5','Class_6',
						'Class_7','Class_8','Class_9'))
		idx = 1
		for c in y_list:
			# print c
			y_bin = np.zeros(9).astype('int')
			y_bin[c] = 1

			row_list = y_bin.tolist()
			row_list.insert(0,idx)
			writer.writerow(row_list)
			idx += 1

	finally:
		f.close()
	


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


################# PREPROCESSING #######################
def contrast_normalize(X, mu=None, stdev=None):
	'''
	INPUT
		- X : n x d dimensional data
	OUTPUT:
		- Z : n x d dimensional normalized data
		- mu : d dimensional mean vector
		- stdev : d dimensional std vector
	'''
	n = X.shape[0]
	if mu is None:
		mu = np.mean(X,axis=0)

	Z = X - mu

	if stdev is None:
		stdev = np.std(Z,axis=0)

	Z = np.divide(Z,np.tile(stdev, (n, 1)))
	return Z, mu, stdev

	# check 'Inf' and 'Nan': should be implemented
	# print np.where(Z == np.nan)
	# print np.where(Z == np.inf)

def zeroone_normalize(X):
	n = X.shape[0]
	max_val = np.max(X, axis=0)
	Z = np.divide(X, np.tile(max_val, (n,1)))
	# print max_val.shape
	# print max_val
	return Z

