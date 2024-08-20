from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

# abstract class python 2 & 3 compatible
ABC = ABCMeta('ABC', (object,), {})

import tensorflow as tf
import numpy as np

import sys
from time import time
from os import makedirs, path
from inspect import isgenerator

from psychrnn.backend.rnn import RNN
# from psychrnn.backend.regularizations import Regularizer
from psychrnn.backend.loss_functions import LossFunction
from psychrnn.backend.initializations import GaussianSpectralRadius, WeightInitializer
from warnings import warn

tf.compat.v1.disable_eager_execution()

class MyBumpBasic(RNN):
    """ The basic continuous time recurrent neural network model.

    Basic implementation of :class:`psychrnn.backend.rnn.RNN` with a simple RNN, enabling biological constraints.

    Args:
       params (dict): See :class:`psychrnn.backend.rnn.RNN` for details.

    """

    def transform(self, x, bs):
        x = x-bs
        u = tf.where(x<1, x,0)*tf.where(x>0, x,0)+tf.where(x>1, tf.sqrt(x*4-3),0)
        u = u+bs

        return u
    
    def normal(self, data, base):
        return data/(1-tf.reduce_mean(base)+tf.reduce_mean(data))

    def recurrent_timestep(self, rnn_in, state):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (*tf.Tensor(dtype=float, shape=(?*, :attr:`N_in` *))*): Input to the rnn at a certain time point.
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at previous time point.

        Returns:
            new_state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): New state of the network.

        """

        new_state = ((1-self.alpha) * state) \
                    + self.alpha * (
                        self.transform( tf.matmul(
                            self.transfer_function(state),
                            self.get_effective_W_rec(),
                            transpose_b=True, name="1")
                        + tf.matmul(
                            rnn_in,
                            self.get_effective_W_in(),
                            transpose_b=True, name="2")
                        + self.b_rec, state))\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                      * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)

        return self.normal(new_state, self.bump_base)

    def recurrent_timestep_original(self, rnn_in, state):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (*tf.Tensor(dtype=float, shape=(?*, :attr:`N_in` *))*): Input to the rnn at a certain time point.
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at previous time point.

        Returns:
            new_state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): New state of the network.

        """

        new_state = ((1-self.alpha) * state) \
                    + self.alpha * (
                        tf.matmul(
                            self.transfer_function(state),
                            self.get_effective_W_rec(),
                            transpose_b=True, name="1")
                        + tf.matmul(
                            rnn_in,
                            self.get_effective_W_in(),
                            transpose_b=True, name="2")
                        + self.b_rec)\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                      * tf.random.normal(tf.shape(input=state), mean=0.0, stddev=1.0)

        return new_state
    
    def output_timestep(self, state):
        """Returns the output node activity for a given timestep.

        Arguments:
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at a given timepoint for each trial in the batch.

        Returns:
            output (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_out` *))*): Output of the network at a given timepoint for each trial in the batch.

        """
     
        output = tf.matmul(self.transfer_function(state),
                                self.get_effective_W_out(), transpose_b=True, name="3") \
                    + self.b_out
     
        return output

    def forward_pass(self):

        """ Run the RNN on a batch of task inputs.

        Iterates over timesteps, running the :func:`recurrent_timestep` and :func:`output_timestep`

        Implements :func:`psychrnn.backend.rnn.RNN.forward_pass`.

        Returns:
            tuple:
            * **predictions** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Network output on inputs found in self.x within the tf network.
            * **states** (*tf.Tensor(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- State variable values over the course of the trials found in self.x within the tf network.

        """
        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        self.bump_base = self.init_state
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            state = self.recurrent_timestep(rnn_input, state)
            output = self.output_timestep(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
        return tf.transpose(a=rnn_outputs, perm=[1, 0, 2]), tf.transpose(a=rnn_states, perm=[1, 0, 2])
