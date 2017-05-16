import math
import numpy as np
import theano

ny = 11

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def OneHot(X, n=None, negative_class=0.):
	print "neg class"
	print n
	X = np.asarray(X).flatten()
	print (X)
	if n is None:
	    n = np.max(X) + 1

	Xoh = np.ones((len(X), n)) * negative_class
	Xoh[np.arange(len(X)), X] = 1.
	return Xoh

def get_buffer_y(steps,num_buffer_samples=10,num_buffer_steps=2):
	num_buffer_rows = int(math.ceil(float(num_buffer_samples) / steps))

	print("num buf rows= " + str(num_buffer_rows))
	targets = np.asarray([[int(round(i*steps+_/num_buffer_steps)) for _ in range(steps)] for i in range(num_buffer_rows)])

	end = np.empty([num_buffer_rows, steps], dtype='int')

	for _ in range( num_buffer_rows):
		empty = np.empty(steps, dtype='int')
		for i in range(steps):
	  		end[_][i] = (int(round(_*steps+i/num_buffer_steps)))


	ymb = floatX(OneHot(end.flatten(), ny))

	return ymb


y = get_buffer_y(5)
print y
