# Code from https://www.kaggle.com/code/tariklemkadem/time-series-classsification
from pandas import read_csv
import numpy as np

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, sep='\s+')
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded
#../input/ucihar-dataset/UCI-HAR Dataset/train/Inertial Signals/total_acc_x_train.txt
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI-HAR Dataset/')

	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI-HAR Dataset/')

	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	print(np.unique(trainy))
	return trainX, trainy, testX, testy

X_train, y_train, X_test, y_test = load_dataset()
np.savez_compressed("UCIHAR.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

