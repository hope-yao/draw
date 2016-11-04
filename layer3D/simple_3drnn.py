'''simple RNN in blocks'''
import numpy
import theano
from theano import tensor
from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks.recurrent import SimpleRecurrent

x = tensor.tensor3('x')
rnn = SimpleRecurrent(dim=4, activation=Identity(), weights_init=initialization.Identity())
rnn.initialize()
h = rnn.apply(x)
f = theano.function([x], h)
print(f(numpy.ones((5, 1, 4), dtype=theano.config.floatX)))
