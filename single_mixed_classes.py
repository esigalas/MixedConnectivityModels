
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


class MixedBrainRegionsRegularizer(object):
    """Regularizer Class

    Class that aggregates all types of regularization used.

    Args:
       params (dict): The regularization parameters containing the following optional keys:

           :Dictionary Keys:
                * **L1_in** (*float, optional*) -- Parameter for weighting the L1 input weights regularization. Default: 0.
                * **L1_rec** (*float, optional*) -- Parameter for weighting the L1 recurrent weights regularization. Default: 0.
                * **L1_out** (*float, optional*) -- Parameter for weighting the L1 output weights regularization. Default: 0.
                * **L2_in** (*float, optional*) -- Parameter for weighting the L2 input weights regularization. Default: 0.
                * **L2_rec** (*float, optional*) -- Parameter for weighting the L2 recurrent weights regularization. Default: 0.
                * **L2_out** (*float, optional*) -- Parameter for weighting the L2 output weights regularization. Default: 0.
                * **L2_firing_rate** (*float, optional*) -- Parameter for weighting the L2 regularization of the relu thresholded states. Default: 0.
                * **custom_regularization** (*function, optional*) -- Custom regularization function. Default: None.

                    Args:
                        * **model** (:class:`~psychrnn.backend.rnn.RNN` *object*) -- Model for which to calculate the regularization.
                        * **params** (*dict*) -- Regularization parameters. All params passed to the :class:`Regularizer` will be passed here.

                    Returns:
                        tf.Tensor(dtype=float)-- The custom regularization to add when calculating the loss.
    """

    def __init__(self, params):
        # ----------------------------------
        # regularization coefficients
        # ----------------------------------
        self.L1_in = params.get('L1_in', 0)
        self.L1_rec = params.get('L1_rec', 0)
        self.L1_out = params.get('L1_out', 0)

        self.L2_in = params.get('L2_in', 0)
        self.L2_rec = params.get('L2_rec', 0)
        self.L2_out = params.get('L2_out', 0)

        self.L2_firing_rate = params.get('L2_firing_rate', 0)

        self.custom_regularization =  params.get('custom_regularization', None)
        self.params = params

    def set_model_regularization(self, model):
        """ Given model, calculate the regularization by adding all regualarization terms (scaled with the parameters to be either zero or nonzero).

        The following regularizations are added: :func:`L1_weight_reg`, :func:`L2_weight_reg`, and :func:`L2_firing_rate_reg`.

        Args:
            model (:class:`~psychrnn.backend.rnn.RNN` object): Model for which to calculate the regularization.

        Returns:
            tf.Tensor(dtype=float): The regularization to add when calculating the loss.
        """
        reg = 0

        # ----------------------------------
        # L1 weight regularization
        # ----------------------------------
        reg += self.L1_weight_reg(model)

        # ----------------------------------
        # L2 weight regularization
        # ----------------------------------
        reg += self.L2_weight_reg(model)

        # ----------------------------------
        # L2 firing rate regularization
        # ----------------------------------
        reg += self.L2_firing_rate_reg(model)


        if self.custom_regularization is not None:
            reg += self.custom_regularization(model, self.params)

        return reg

    def L1_weight_reg(self, model):
        """ L1 regularization

        :math:`regularization = L1\\_in * ||W\\_in||_1 + L1\\_rec * ||W\\_rec||_1 + L1\\_out * ||W\\_out||_1`

        Args:
            model (:class:`~psychrnn.backend.rnn.RNN` object): Model for which to calculate the regularization.

        Returns:
            tf.Tensor(dtype=float): The L1 regularization to add when calculating the loss.
        """

        reg = 0

        reg += self.L1_in * tf.reduce_mean(input_tensor=tf.abs(model.get_effective_W_in1_1()))
        reg += self.L1_rec * tf.reduce_mean(input_tensor=tf.abs(model.get_effective_W_rec1_1()))
        reg += self.L1_out * tf.reduce_mean(input_tensor=tf.abs(model.get_effective_W_out()))

        return reg

    def L2_weight_reg(self, model):
        """ L2 regularization

        :math:`regularization = L2\\_in * ||W\\_in||_2^2 + L2\\_rec * ||W\\_rec||_2^2 + L2\\_out * ||W\\_out||_2^2`

        Args:
            model (:class:`~psychrnn.backend.rnn.RNN` object): Model for which to calculate the regularization.

        Returns:
            tf.Tensor(dtype=float): The L2 regularization to add when calculating the loss.
        """

        reg = 0

        reg += self.L2_in * tf.reduce_mean(input_tensor=tf.square(tf.abs(model.get_effective_W_in1_1())))
        reg += self.L2_rec * tf.reduce_mean(input_tensor=tf.square(tf.abs(model.get_effective_W_rec1_1())))
        reg += self.L2_out * tf.reduce_mean(input_tensor=tf.square(tf.abs(model.get_effective_W_out())))

        return reg

    def L2_firing_rate_reg(self, model):
        """ L2 regularization of the firing rate.

        :math:`regularization = L2\\_firing\\_rate * ||relu(states)||_2^2`

        Args:
            model (:class:`~psychrnn.backend.rnn.RNN` object): Model for which to calculate the regularization.

        Returns:
            tf.Tensor(dtype=float): The L2 firing rate regularization to add when calculating the loss.
        """
        reg = self.L2_firing_rate * tf.reduce_mean(input_tensor=tf.square(model.transfer_function(model.states)))

        return reg


class MixedBrainRegionsWeightInitializer(object):
    """ Base Weight Initialization class.

    Initializes biological constraints and network weights, optionally loading weights from a file or from passed in arrays.

    Keyword Arguments:
        N_in (int): The number of network inputs.
        N_rec (int): The number of recurrent units in the network.
        N_out (int): The number of network outputs.
        load_weights_path (str, optional): Path to load weights from using np.load. Weights saved at that path should be in the form saved out by :func:`psychrnn.backend.rnn.RNN.save` Default: None.
        
        input_connectivity (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in`)), optional): Connectivity mask for the input layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_rec`, :attr:`N_in`)).
        rec_connectivity (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`)), optional): Connectivity mask for the recurrent layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_rec`, :attr:`N_rec`)).
        output_connectivity (ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec`)), optional): Connectivity mask for the output layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_out`, :attr:`N_rec`)).
        autapses (bool, optional): If False, self connections are not allowed in N_rec, and diagonal of :data:`rec_connectivity` will be set to 0. Default: True.
        dale_ratio (float, optional): Dale's ratio, used to construct Dale_rec and Dale_out. 0 <= dale_ratio <=1 if dale_ratio should be used. ``dale_ratio * N_rec`` recurrent units will be excitatory, the rest will be inhibitory. Default: None
        
        which_rand_init (str, optional): Which random initialization to use for W_in and W_out. Will also be used for W_rec if :data:`which_rand_W_rec_init` is not passed in. Options: :func:`'const_unif' <const_unif_init>`, :func:`'const_gauss' <const_gauss_init>`, :func:`'glorot_unif' <glorot_unif_init>`, :func:`'glorot_gauss' <glorot_gauss_init>`. Default: :func:`'glorot_gauss' <glorot_gauss_init>`.
        which_rand_W_rec_init (str, optional): Which random initialization to use for W_rec. Options: :func:`'const_unif' <const_unif_init>`, :func:`'const_gauss' <const_gauss_init>`, :func:`'glorot_unif' <glorot_unif_init>`, :func:`'glorot_gauss' <glorot_gauss_init>`. Default: :data:`which_rand_init`.
        init_minval (float, optional): Used by :func:`const_unif_init` as :attr:`minval` if ``'const_unif'`` is passed in for :data:`which_rand_init` or :data:`which_rand_W_rec_init`. Default: -.1.
        init_maxval (float, optional): Used by :func:`const_unif_init` as :attr:`maxval` if ``'const_unif'`` is passed in for :data:`which_rand_init` or :data:`which_rand_W_rec_init`. Default: .1. 

        W_in (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in` )), optional): Input weights. Default: Initialized using the function indicated by :data:`which_rand_init`
        W_rec (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` )), optional): Recurrent weights. Default: Initialized using the function indicated by :data:`which_rand_W_rec_init`
        W_out (ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec` )), optional): Output weights. Defualt: Initialized using the function indicated by :data:`which_rand_init`
        b_rec (ndarray(dtype=float, shape=(:attr:`N_rec`, )), optional): Recurrent bias. Default: np.zeros(:attr:`N_rec`)
        b_out (ndarray(dtype=float, shape=(:attr:`N_out`, )), optional): Output bias. Default: np.zeros(:attr:`N_out`)
        Dale_rec (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`)), optional): Diagonal matrix with ones and negative ones on the diagonal. If :data:`dale_ratio` is not ``None``, indicates whether a recurrent unit is excitatory(1) or inhibitory(-1). Default: constructed based on :data:`dale_ratio`
        Dale_out (ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`)), optional): Diagonal matrix with ones and zeroes on the diagonal. If :data:`dale_ratio` is not ``None``, indicates whether a recurrent unit is excitatory(1) or inhibitory(0). Inhibitory neurons do not contribute to the output. Default: constructed based on :data:`dale_ratio`
        init_state (ndarray(dtype=float, shape=(1, :attr:`N_rec` )), optional): Initial state of the network's recurrent units. Default: .1 + .01 * np.random.randn(:data:`N_rec` ).

    Attributes:
        initializations (dict): Dictionary containing entries for :data:`input_connectivity`, :data:`rec_connectivity`, :data:`output_connectivity`, :data:`dale_ratio`, :data:`Dale_rec`, :data:`Dale_out`, :data:`W_in`, :data:`W_rec`, :data:`W_out`, :data:`b_rec`, :data:`b_out`, and :data:`init_state`.
    
    """


    def __init__(self, **kwargs):

        # ----------------------------------
        # Required parameters
        # ----------------------------------
        
        self.load_weights_path = kwargs.get('load_weights_path', None)
        
        N_in = self.N_in = kwargs.get('N_in')
        
        N_rec1_1 = self.N_rec1_1 = kwargs.get('N_rec1_1')
        N_rec1_2 = self.N_rec1_2 = kwargs.get('N_rec1_2')

        N_rec = self.N_rec = kwargs.get('N_rec')
        N_out = self.N_out = kwargs.get('N_out')
        self.autapses = kwargs.get('autapses', True)

        self.initializations = dict()

        if self.load_weights_path is not None:
            # ----------------------------------
            # Load saved weights
            # ----------------------------------
            self.initializations = dict(np.load(self.load_weights_path, allow_pickle = True))
            if 'dale_ratio' in self.initializations.keys():
                if type(self.initializations['dale_ratio']) == np.ndarray:
                    self.initializations['dale_ratio'] = self.initializations['dale_ratio'].item()
            else:
                warn("You are loading weights from a model trained with an old version (<1.0). Dale's formatting has changed. Dale's rule will not be applied even if the model was previously trained using Dale's. To change this behavior, add the correct dale ratio to the 'dale_ratio' field to the file that weights are being loaded from, " + self.load_weights_path + ".")
                self.initializations['dale_ratio']  = None;

        else:

            if kwargs.get('W_rec1_1', None) is None and type(self).__name__=='MixedBrainRegionsWeightInitializer':
                warn("This network may not train since the eigenvalues of W_rec1_1 are not regulated in any way.")
            if kwargs.get('W_rec1_2', None) is None and type(self).__name__=='MixedBrainRegionsWeightInitializer':
                warn("This network may not train since the eigenvalues of W_rec1_2 are not regulated in any way.")

            # ----------------------------------
            # Optional Parameters
            # ----------------------------------
            self.rand_init = kwargs.get('which_rand_init', 'glorot_gauss')
            self.rand_W_rec_init = self.get_rand_init_func(kwargs.get('which_rand_W_rec_init', self.rand_init))
            self.rand_init = self.get_rand_init_func(self.rand_init)
            if self.rand_init == self.const_unif_init or self.rand_W_rec_init == self.const_unif_init:
                self.init_minval = kwargs.get('init_minval', -.01)
                self.init_maxval = kwargs.get('init_maxval', .01)

            # ----------------------------------
            # Biological Constraints
            # ----------------------------------

            # Connectivity constraints
            self.initializations['input_connectivity1_1'] = kwargs.get('input_connectivity1_1',np.ones([N_rec1_1, N_in+N_rec1_2]))
            assert(self.initializations['input_connectivity1_1'].shape == (N_rec1_1, N_in+N_rec1_2))
            self.initializations['input_connectivity1_2'] = kwargs.get('input_connectivity1_2',np.ones([N_rec1_2, N_in+N_rec1_1]))
            assert(self.initializations['input_connectivity1_2'].shape == (N_rec1_2, N_in+N_rec1_1))

            self.initializations['rec_connectivity1_1'] = kwargs.get('rec_connectivity1_1',np.ones([N_rec1_1, N_rec1_1]))
            assert(self.initializations['rec_connectivity1_1'].shape == (N_rec1_1, N_rec1_1))
            self.initializations['rec_connectivity1_2'] = kwargs.get('rec_connectivity1_2',np.ones([N_rec1_2, N_rec1_2]))
            assert(self.initializations['rec_connectivity1_2'].shape == (N_rec1_2, N_rec1_2))
            
            self.initializations['output_connectivity'] = kwargs.get('output_connectivity', np.ones([N_out, N_rec1_1+N_rec1_2]))
            assert(self.initializations['output_connectivity'].shape == (N_out, N_rec1_1+N_rec1_2))
            
            # Autapses constraint
            if not self.autapses:
                self.initializations['rec_connectivity1_1'][np.eye(N_rec1_1) == 1] = 0
                self.initializations['rec_connectivity1_2'][np.eye(N_rec1_2) == 1] = 0

            # Dale's constraint
            self.initializations['dale_ratio'] = dale_ratio = kwargs.get('dale_ratio', None)
            if type(self.initializations['dale_ratio']) == np.ndarray:
                self.initializations['dale_ratio'] = dale_ratio = self.initializations['dale_ratio'].item()
            if dale_ratio is not None and (dale_ratio <0 or dale_ratio > 1):
                print("Need 0 <= dale_ratio <= 1. dale_ratio was: " + str(dale_ratio))
                raise
            dale_vec = np.ones(N_rec)
            if dale_ratio is not None:
                dale_vec[int(dale_ratio * N_rec):] = -1
                dale_rec = np.diag(dale_vec)
                dale_vec[int(dale_ratio * N_rec):] = 0
                dale_out = np.diag(dale_vec)
            else:
                dale_rec = np.diag(dale_vec)
                dale_out = np.diag(dale_vec)
            self.initializations['Dale_rec'] = kwargs.get('Dale_rec', dale_rec)
            assert(self.initializations['Dale_rec'].shape == (N_rec, N_rec))
            self.initializations['Dale_out'] = kwargs.get('Dale_out', dale_rec)
            assert(self.initializations['Dale_out'].shape == (N_rec, N_rec))

            # ----------------------------------
            # Default initializations / optional loading from params
            # ----------------------------------
            self.initializations['W_in1_1'] = kwargs.get('W_in1_1', self.rand_init(self.initializations['input_connectivity1_1']))
            assert(self.initializations['W_in1_1'].shape == (N_rec1_1, N_in+N_rec1_2))
            self.initializations['W_in1_2'] = kwargs.get('W_in1_2', self.rand_init(self.initializations['input_connectivity1_2']))
            assert(self.initializations['W_in1_2'].shape == (N_rec1_2, N_in+N_rec1_1))
            
            self.initializations['W_out'] = kwargs.get('W_out', self.rand_init(self.initializations['output_connectivity']))
            assert(self.initializations['W_out'].shape == (N_out, N_rec1_1+N_rec1_2))

            self.initializations['W_rec1_1'] = kwargs.get('W_rec1_1', self.rand_W_rec_init(self.initializations['rec_connectivity1_1']))
            assert(self.initializations['W_rec1_1'].shape == (N_rec1_1, N_rec1_1))
            self.initializations['W_rec1_2'] = kwargs.get('W_rec1_2', self.rand_W_rec_init(self.initializations['rec_connectivity1_2']))
            assert(self.initializations['W_rec1_2'].shape == (N_rec1_2, N_rec1_2))
            
            self.initializations['b_rec1_1'] = kwargs.get('b_rec1_1',np.zeros(N_rec1_1))
            assert(self.initializations['b_rec1_1'].shape == (N_rec1_1,))
            self.initializations['b_rec1_2'] = kwargs.get('b_rec1_2',np.zeros(N_rec1_2))
            assert(self.initializations['b_rec1_2'].shape == (N_rec1_2,))
            
            self.initializations['b_out'] = kwargs.get('b_out',np.zeros(N_out))
            assert(self.initializations['b_out'].shape == (N_out,))

            self.initializations['init_state1_1'] = kwargs.get('init_state1_1', .01 + .01 * np.random.randn(N_rec1_1))
            assert(self.initializations['init_state1_1'].size == N_rec1_1)

            self.initializations['init_state1_2'] = kwargs.get('init_state1_2', .01 + .01 * np.random.randn(N_rec1_2))
            assert(self.initializations['init_state1_2'].size == N_rec1_2)

        return

    def get_rand_init_func(self, which_rand_init):
        """Maps initialization function names (strings) to generating functions.

        Arguments:
            which_rand_init (str): Maps to ``[which_rand_init]_init``. Options are :func:`'const_unif' <const_unif_init>`, :func:`'const_gauss' <const_gauss_init>`, :func:`'glorot_unif' <glorot_unif_init>`, :func:`'glorot_gauss' <glorot_gauss_init>`.

        Returns:
            function: ``self.[which_rand_init]_init``

        """
        mapping = {
            'const_unif': self.const_unif_init,
            'const_gauss': self.const_gauss_init,
            'glorot_unif': self.glorot_unif_init,
            'glorot_gauss': self.glorot_gauss_init}
        return mapping[which_rand_init]

    def const_gauss_init(self, connectivity):
        """ Initialize ndarray of shape :data:`connectivity` with values from a normal distribution.

        Arguments:
            connectivity (ndarray): 1 where connected, 0 where unconnected.

        Returns:
            ndarray(dtype=float, shape=connectivity.shape)

        """
        return np.random.randn(connectivity.shape[0], connectivity.shape[1])

    def const_unif_init(self, connectivity):
        """ Initialize ndarray of shape :data:`connectivity` with values uniform distribution with minimum :data:`init_minval` and maximum :data:`init_maxval` as set in :class:`MixedWeightInitializer`.

        Arguments:
            connectivity (ndarray): 1 where connected, 0 where unconnected.

        Returns:
            ndarray(dtype=float, shape=connectivity.shape)

        """
        minval = self.init_minval
        maxval = self.init_maxval
        return (maxval-minval) * np.random.rand(connectivity.shape[0], connectivity.shape[1]) + minval

    def glorot_unif_init(self, connectivity):
        """ Initialize ndarray of shape :data:`connectivity` with values from a glorot uniform distribution.

        Draws samples from a uniform distribution within [-limit, limit] where `limit`
        is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units and `fan_out` is the number of output units. Respects the :data:`connectivity` matrix.

        Arguments:
            connectivity (ndarray): 1 where connected, 0 where unconnected.

        Returns:
            ndarray(dtype=float, shape=connectivity.shape)

        """


        init = np.zeros(connectivity.shape)
        fan_in = np.sum(connectivity, axis = 1)
        init += np.tile(fan_in, (connectivity.shape[1],1)).T
        fan_out = np.sum(connectivity, axis = 0)
        init += np.tile(fan_out, (connectivity.shape[0],1))
        return np.random.uniform(-np.sqrt(6/init), np.sqrt(6/init))

    def glorot_gauss_init(self, connectivity):
        """ Initialize ndarray of shape :data:`connectivity` with values from a glorot normal distribution.

        Draws samples from a normal distribution centered on 0 with `stddev
        = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of input units and `fan_out` is the number of output units. Respects the :data:`connectivity` matrix.

        Arguments:
            connectivity (ndarray): 1 where connected, 0 where unconnected.

        Returns:
            ndarray(dtype=float, shape=connectivity.shape)

        """

        init = np.zeros(connectivity.shape)
        fan_in = np.sum(connectivity, axis = 1)
        init += np.tile(fan_in, (connectivity.shape[1],1)).T
        fan_out = np.sum(connectivity, axis = 0)
        init += np.tile(fan_out, (connectivity.shape[0],1))
        return np.random.normal(0, np.sqrt(2/init))

    def get_dale_ratio(self):
        """ Returns the dale_ratio.

        :math:`0 \\leq dale\\_ratio \\leq 1` if dale_ratio should be used, dale_ratio = None otherwise. ``dale_ratio * N_rec`` recurrent units will be excitatory, the rest will be inhibitory.

        Returns:
            float: Dale ratio, None if no dale ratio is set.
        """

        return self.initializations['dale_ratio']

    def get(self, tensor_name):
        """ Get :data:`tensor_name` from :attr:`initializations` as a Tensor.

        Arguments:
            tensor_name (str): The name of the tensor to get. See :attr:`initializations` for options.

        Returns:
            Tensor object

        """

        return tf.compat.v1.constant_initializer(self.initializations[tensor_name])

    def save(self, save_path):
        """ Save :attr:`initializations` to :data:`save_path`.

        Arguments:
            save_path (str): File path for saving the initializations. The .npz extension will be appended if not already provided.
        """

        np.savez(save_path, **self.initializations)
        return

    def balance_dale_ratio(self):
        """ If dale_ratio is not None, balances :attr:`initializations['W_rec'] <initializations>` 's excitatory and inhibitory weights so the network will train. 
        """
        dale_ratio = self.get_dale_ratio()
        if dale_ratio is not None:
            dale_vec = np.ones(self.N_rec)
            dale_vec[int(dale_ratio * self.N_rec):] = dale_ratio/(1-dale_ratio)
            dale_rec = np.diag(dale_vec) / np.linalg.norm(np.matmul(self.initializations['rec_connectivity'], np.diag(dale_vec)), axis=1)[:,np.newaxis]
            self.initializations['W_rec'] = np.matmul(self.initializations['W_rec'], dale_rec)
        return


class mixedBrainRegionsRNN(ABC):
    """ The base recurrent neural network class.

    Note:
        The base RNN class is not itself a functioning RNN. 
        forward_pass must be implemented to define a functioning RNN.

    Args:
       params (dict): The RNN parameters. Use your tasks's :func:`~psychrnn.tasks.task.Task.get_task_params` function to start building this dictionary. Optionally use a different network's :func:`get_weights` function to initialize the network with preexisting weights.

       :Dictionary Keys: 
            * **name** (*str*) -- Unique name used to determine variable scope. Having different variable scopes allows multiple distinct models to be instantiated in the same TensorFlow environment. See `TensorFlow's variable_scope <https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope>`_ for more details.
            * **N_in** (*int*) -- The number of network inputs.
            * **N_rec** (*int*) -- The number of recurrent units in the network.
            * **N_out** (*int*) -- The number of network outputs.
            * **N_steps** (*int*): The number of simulation timesteps in a trial. 
            * **dt** (*float*) -- The simulation timestep.
            * **tau** (*float*) -- The intrinsic time constant of neural state decay.
            * **N_batch** (*int*) -- The number of trials per training update.
            * **rec_noise** (*float, optional*) -- How much recurrent noise to add each time the new state of the network is calculated. Default: 0.0.
            * **transfer_function** (*function, optional*) -- Transfer function to use for the network. Default: `tf.nn.relu <https://www.tensorflow.org/api_docs/python/tf/nn/relu>`_.
            * **load_weights_path** (*str, optional*) -- When given a path, loads weights from file in that path. Default: None
            * **initializer** (:class:`~psychrnn.backend.initializations.WeightInitializer` *or child object, optional*) -- Initializer to use for the network. Default: :class:`~psychrnn.backend.initializations.WeightInitializer` (:data:`params`) if :data:`params` includes :data:`W_rec` or :data:`load_weights_path` as a key, :class:`~psychrnn.backend.initializations.GaussianSpectralRadius` (:data:`params`) otherwise.
            * **W_in_train** (*bool, optional*) -- True if input weights, W_in, are trainable. Default: True
            * **W_rec_train** (*bool, optional*) -- True if recurrent weights, W_rec, are trainable. Default: True
            * **W_out_train** (*bool, optional*) -- True if output weights, W_out, are trainable. Default: True
            * **b_rec_train** (*bool, optional*) -- True if recurrent bias, b_rec, is trainable. Default: True
            * **b_out_train** (*bool, optional*) -- True if output bias, b_out, is trainable. Default: True
            * **init_state_train** (*bool, optional*) -- True if the inital state for the network, init_state, is trainable. Default: True
            * **loss_function** (*str, optional*) -- Which loss function to use. See :class:`psychrnn.backend.loss_functions.LossFunction` for details. Defaults to ``"mean_squared_error"``.



        :Other Dictionary Keys:
            * Any dictionary keys used by the regularizer will be passed onwards to :class:`psychrnn.backend.regularizations.Regularizer`. See :class:`~psychrnn.backend.regularizations.Regularizer` for key names and details.
            * Any dictionary keys used for the loss function will be passed onwards to :class:`psychrnn.backend.loss_functions.LossFunction`. See :class:`~psychrnn.backend.loss_functions.LossFunction` for key names and details.
            * If :data:`initializer` is not set, any dictionary keys used by the initializer will be pased onwards to :class:`WeightInitializer <psychrnn.backend.initializations.WeightInitializer>` if :data:`load_weights_path` is set or :data:`W_rec` is passed in. Otherwise all keys will be passed to :class:`GaussianSpectralRadius <psychrnn.backend.initializations.GaussianSpectralRadius>`
            * If :data:`initializer` is not set and :data:`load_weights_path` is not set, the dictionary entries returned previously by :func:`get_weights` can be passed in to initialize the network. See :class:`WeightInitializer <psychrnn.backend.initializations.WeightInitializer>` for a list and explanation of possible parameters. At a minimum, :data:`W_rec` must be included as a key to make use of this option.
            * If :data:`initializer` is not set and :data:`load_weights_path` is not set, the following keys can be used to set biological connectivity constraints:

                * **input_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_rec`, :attr:`N_in` *)), optional*) -- Connectivity mask for the input layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_rec`, :attr:`N_in`)).
                * **rec_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_rec`, :attr:`N_rec` *)), optional*) -- Connectivity mask for the recurrent layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_rec`, :attr:`N_rec`)).
                * **output_connectivity** (*ndarray(dtype=float, shape=(* :attr:`N_out`, :attr:`N_rec` *)), optional*) -- Connectivity mask for the output layer. 1 where connected, 0 where unconnected. Default: np.ones((:attr:`N_out`, :attr:`N_rec`)).
                * **autapses** (*bool, optional*) -- If False, self connections are not allowed in N_rec, and diagonal of :data:`rec_connectivity` will be set to 0. Default: True.
                * **dale_ratio** (float, optional) -- Dale's ratio, used to construct Dale_rec and Dale_out. 0 <= dale_ratio <=1 if dale_ratio should be used. ``dale_ratio * N_rec`` recurrent units will be excitatory, the rest will be inhibitory. Default: None
        
        Inferred Parameters:
            * **alpha** (*float*) -- The number of unit time constants per simulation timestep.

    """
    def __init__(self, params):
        self.params = params

        # --------------------------------------------
        # Unique name used to determine variable scope
        # --------------------------------------------
        try:
            self.name = params['name']
        except KeyError:
            print("You must pass a  'name' to RNN")
            raise

        # ----------------------------------
        # Network sizes (tensor dimensions)
        # ----------------------------------
        try:
            N_in = self.N_in = params['N_in']
        except KeyError:
            print("You must pass 'N_in' to RNN")
            raise
        try:
            N_rec1_1 = self.N_rec1_1 = params['N_rec1_1']
        except KeyError:
            print("You must pass 'N_rec1_1' to RNN")
            raise
        try:
            N_out = self.N_out = params['N_out']
        except KeyError:
            print("You must pass 'N_out' to RNN")
            raise
        try:
            N_steps = self.N_steps = params['N_steps']
        except KeyError:
            print("You must pass 'N_steps' to RNN")
            raise
        try:
            N_rec1_2 = self.N_rec1_2 = params['N_rec1_2']
        except KeyError:
            print("You must pass 'N_rec1_2' to RNN")
            raise
        try:
            N_rec = self.N_rec = params['N_rec']
        except KeyError:
            print("You must pass 'N_rec' to RNN")
            raise

        # ----------------------------------
        # Physical parameters
        # ----------------------------------
        try:
            self.dt = params['dt']
        except KeyError:
            print("You must pass 'dt' to RNN")
            raise
            
        try:
            self.tau = params['tau']
        except KeyError:
            print("You must pass 'tau' to RNN")
            raise
        try: 
            self.tau = self.tau.astype('float32')
        except AttributeError:
            pass

        try:
            self.N_batch = params['N_batch']
        except KeyError:
            print("You must pass 'N_batch' to RNN")
            raise
            
        self.alpha = (1.0 * self.dt) / self.tau
        self.rec_noise1 = params.get('rec_noise1', 0.0)
        self.transfer_function = params.get('transfer_function', tf.nn.relu)

        # ----------------------------------
        # Load weights path
        # ----------------------------------
        self.load_weights_path = params.get('load_weights_path', None)

        # ------------------------------------------------
        # Define initializer for TensorFlow variables
        # ------------------------------------------------
        if self.load_weights_path is not None:
            self.initializer = MixedBrainRegionsWeightInitializer(load_weights_path=self.load_weights_path)
        elif params.get('W_rec1_1', None) is not None:
            self.initializer = params.get('initializer',
                                          MixedBrainRegionsWeightInitializer(**params))
        else:
            self.initializer = params.get('initializer',
                                          GaussianSpectralRadius(**params))

        self.dale_ratio = self.initializer.get_dale_ratio()

        # ----------------------------------
        # Trainable features
        # ----------------------------------
        self.W_in_train1_1 = params.get('W_in_train1_1', True)
        self.W_rec_train1_1 = params.get('W_rec_train1_1', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train1_1 = params.get('b_rec_train1_1', True)
        self.b_out_train = params.get('b_out_train', True)
        self.init_state_train = params.get('init_state_train', True)
        
        self.W_in_train1_2 = params.get('W_in_train1_2', True)
        self.W_rec_train1_2 = params.get('W_rec_train1_2', True)
        self.b_rec_train1_2 = params.get('b_rec_train1_2', True)

        # --------------------------------------------------
        # TensorFlow input/output placeholder initializations
        # ---------------------------------------------------
        self.x = tf.compat.v1.placeholder("float", [None, N_steps, N_in])
        self.y = tf.compat.v1.placeholder("float", [None, N_steps, N_out])
        self.output_mask = tf.compat.v1.placeholder("float", [None, N_steps, N_out])

        # --------------------------------------------------
        # Initialize variables in proper scope
        # ---------------------------------------------------
        with tf.compat.v1.variable_scope(self.name) as scope:
            # ------------------------------------------------
            # Trainable variables:
            # Initial State, weight matrices and biases
            # ------------------------------------------------
            try:
                self.init_state1_1 = tf.compat.v1.get_variable('init_state1_1', [1, N_rec1_1],
                                              initializer=self.initializer.get('init_state1_1'),
                                              trainable=self.init_state_train)
                self.init_state1_2 = tf.compat.v1.get_variable('init_state1_2', [1, N_rec1_2],
                                              initializer=self.initializer.get('init_state1_2'),
                                              trainable=self.init_state_train)
            except ValueError as error:
                raise UserWarning("Try calling model.destruct() or changing params['name'].")


            self.init_state1_1 = tf.tile(self.init_state1_1, [self.N_batch, 1])
            self.init_state1_2 = tf.tile(self.init_state1_2, [self.N_batch, 1])

            # Input weight matrix:
            self.W_in1_1 = \
                tf.compat.v1.get_variable('W_in1_1', [N_rec1_1, N_in+N_rec1_2],
                                initializer=self.initializer.get('W_in1_1'),
                                trainable=self.W_in_train1_1)
            self.W_in1_2 = \
                tf.compat.v1.get_variable('W_in1_2', [N_rec1_2, N_in+N_rec1_1],
                                initializer=self.initializer.get('W_in1_2'),
                                trainable=self.W_in_train1_2)

            # Recurrent weight matrix:
            self.W_rec1_1 = \
                tf.compat.v1.get_variable(
                    'W_rec1_1',
                    [N_rec1_1, N_rec1_1],
                    initializer=self.initializer.get('W_rec1_1'),
                    trainable=self.W_rec_train1_1)
            self.W_rec1_2 = \
                tf.compat.v1.get_variable(
                    'W_rec1_2',
                    [N_rec1_2, N_rec1_2],
                    initializer=self.initializer.get('W_rec1_2'),
                    trainable=self.W_rec_train1_2)

            # Output weight matrix:
            self.W_out = tf.compat.v1.get_variable('W_out', [N_out, N_rec1_1+N_rec1_2],
                                         initializer=self.initializer.get('W_out'),
                                         trainable=self.W_out_train)

            # Recurrent bias:
            self.b_rec1_1 = tf.compat.v1.get_variable('b_rec1_1', [N_rec1_1], initializer=self.initializer.get('b_rec1_1'),
                                         trainable=self.b_rec_train1_1)
            self.b_rec1_2 = tf.compat.v1.get_variable('b_rec1_2', [N_rec1_2], initializer=self.initializer.get('b_rec1_2'),
                                         trainable=self.b_rec_train1_2)

            # Output bias:
            self.b_out = tf.compat.v1.get_variable('b_out', [N_out], initializer=self.initializer.get('b_out'),
                                         trainable=self.b_out_train)

            # ------------------------------------------------
            # Non-trainable variables:
            # Overall connectivity and Dale's law matrices
            # ------------------------------------------------

            # Recurrent Dale's law weight matrix:
            self.Dale_rec = tf.compat.v1.get_variable('Dale_rec', [N_rec, N_rec],
                                            initializer=self.initializer.get('Dale_rec'),
                                            trainable=False)

            # Output Dale's law weight matrix:
            self.Dale_out = tf.compat.v1.get_variable('Dale_out', [N_rec, N_rec],
                                            initializer=self.initializer.get('Dale_out'),
                                            trainable=False)

            # Connectivity weight matrices:
            self.input_connectivity1_1 = tf.compat.v1.get_variable('input_connectivity1_1', [N_rec1_1, N_in+N_rec1_2],
                                                      initializer=self.initializer.get('input_connectivity1_1'),
                                                      trainable=False)
            self.input_connectivity1_2 = tf.compat.v1.get_variable('input_connectivity1_2', [N_rec1_2, N_in+N_rec1_1],
                                                      initializer=self.initializer.get('input_connectivity1_2'),
                                                      trainable=False)
        
            self.rec_connectivity1_1 = tf.compat.v1.get_variable('rec_connectivity1_1', [N_rec1_1, N_rec1_1],
                                                    initializer=self.initializer.get('rec_connectivity1_1'),
                                                    trainable=False)
            self.rec_connectivity1_2 = tf.compat.v1.get_variable('rec_connectivity1_2', [N_rec1_2, N_rec1_2],
                                                    initializer=self.initializer.get('rec_connectivity1_2'),
                                                    trainable=False)
        
            self.output_connectivity = tf.compat.v1.get_variable('output_connectivity', [N_out, N_rec1_1+N_rec1_2],
                                                       initializer=self.initializer.get('output_connectivity'),
                                                       trainable=False)

        # --------------------------------------------------
        # Flag to check if variables initialized, model built
        # ---------------------------------------------------
        self.is_initialized = False
        self.is_built = False

    def build(self):
        """ Build the TensorFlow network and start a TensorFlow session.

        """
        # --------------------------------------------------
        # Define the predictions
        # --------------------------------------------------
        self.predictions, self.states = self.forward_pass()

        # --------------------------------------------------
        # Define the loss (based on the predictions)
        # --------------------------------------------------
        self.loss = LossFunction(self.params).set_model_loss(self)

        # --------------------------------------------------
        # Define the regularization
        # --------------------------------------------------
        self.reg = MixedBrainRegionsRegularizer(self.params).set_model_regularization(self)

        # --------------------------------------------------
        # Define the total regularized loss
        # --------------------------------------------------
        self.reg_loss = self.loss + self.reg

        # --------------------------------------------------
        # Open a session
        # --------------------------------------------------
        self.sess = tf.compat.v1.Session()

        # --------------------------------------------------
        # Record successful build
        # --------------------------------------------------
        self.is_built = True

        return

    def destruct(self):
        """ Close the TensorFlow session and reset the global default graph.

        """
        # --------------------------------------------------
        # Close the session. Delete the graph.
        # --------------------------------------------------
        if self.is_built:
            self.sess.close()
        tf.compat.v1.reset_default_graph()
        return

    def get_effective_W_rec1_1(self):
        """ Get the recurrent weights used in the network, after masking by connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` ))

        """
        W_rec = self.W_rec1_1 * self.rec_connectivity1_1
        if self.dale_ratio:
            W_rec = tf.matmul(tf.abs(W_rec), self.Dale_rec, name="in_1_1")
        return W_rec

    def get_effective_W_rec1_2(self):
        """ Get the recurrent weights used in the network, after masking by connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` ))

        """
        W_rec = self.W_rec1_2 * self.rec_connectivity1_2
        if self.dale_ratio:
            W_rec = tf.matmul(tf.abs(W_rec), self.Dale_rec, name="in_1_2")
        return W_rec

    def get_effective_W_in1_1(self):
        """ Get the input weights used in the network, after masking by connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in` ))
        """

        W_in = self.W_in1_1 * self.input_connectivity1_1
        if self.dale_ratio:
            W_in = tf.abs(W_in)
        return W_in

    def get_effective_W_in1_2(self):
        """ Get the input weights used in the network, after masking by connectivity and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in` ))
        """

        W_in = self.W_in1_2 * self.input_connectivity1_2
        if self.dale_ratio:
            W_in = tf.abs(W_in)
        return W_in

    def get_effective_W_out(self):
        """ Get the output weights used in the network, after masking by connectivity, and dale_ratio.

        Returns:
            tf.Tensor(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec` ))
        """

        W_out = self.W_out * self.output_connectivity
        if self.dale_ratio:
            W_out = tf.matmul(tf.abs(W_out), self.Dale_out, name="in_2")
        return W_out

    def transform(self, x, bs):
            x = x-bs
            u = tf.where(x<1, x,0)*tf.where(x>0, x,0)+tf.where(x>1, tf.sqrt(x*4-3),0)
            u = u+bs

            return u

    def normal(self, data, base):
        return data/(1-tf.reduce_mean(base)+tf.reduce_mean(data))

    def recurrent_timestep(self, bump_in1, non_bump_in1, bump_state1, non_bump_state1):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.

        Arguments:
            rnn_in (*tf.Tensor(dtype=float, shape=(?*, :attr:`N_in` *))*): Input to the rnn at a certain time point.
            state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): State of network at previous time point.

        Returns:
            new_state (*tf.Tensor(dtype=float, shape=(* :attr:`N_batch` , :attr:`N_rec` *))*): New state of the network.

        """
        # mixed region update
        new_non_bump_state1 = ((1-self.alpha) * non_bump_state1) \
                    + self.alpha * (
                        tf.matmul(
                            self.transfer_function(non_bump_state1),
                            self.get_effective_W_rec1_2(),
                            transpose_b=True, name="3")
                        + tf.matmul(
                            non_bump_in1,
                            self.get_effective_W_in1_2(),
                            transpose_b=True, name="4")
                        + self.b_rec1_2)\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise1 * self.rec_noise1)\
                      * tf.random.normal(tf.shape(input=non_bump_state1), mean=0.0, stddev=1.0)
        
        new_bump_state1 = ((1-self.alpha) * bump_state1) \
                    + self.alpha * (self.transform(tf.matmul(bump_state1, self.get_effective_W_rec1_1(), transpose_b=True, name="1") \
                                                 + tf.matmul(bump_in1, self.get_effective_W_in1_1(), transpose_b=True, name="2")\
                                                 + self.b_rec1_1,
                                                   bump_state1))\
                    + tf.sqrt(2.0 * self.alpha * self.rec_noise1 * self.rec_noise1)\
                      * tf.random.normal(tf.shape(input=bump_state1), mean=0.0, stddev=1.0)

        return self.normal(new_bump_state1, self.bump_base1), new_non_bump_state1

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

        bump_state1 = self.init_state1_1
        self.bump_base1 = self.init_state1_1

        non_bump_state1 = self.init_state1_2
        self.non_bump_base1 = self.init_state1_2

        rnn_outputs = []

        bump_states1 = []
        non_bump_states1 = []
        
        full_states = []
        for rnn_input in rnn_inputs:
            bump_state1, non_bump_state1 = self.recurrent_timestep(tf.concat([rnn_input, non_bump_state1],1),
                                                                   tf.concat([rnn_input, bump_state1],1),
                                                                   bump_state1, non_bump_state1)

            full_state = tf.concat([bump_state1, non_bump_state1], axis=1)

            output = self.output_timestep(full_state)
            rnn_outputs.append(output)

            bump_states1.append(bump_state1)
            non_bump_states1.append(non_bump_state1)

            full_states.append(full_state)

        return tf.transpose(a=rnn_outputs, perm=[1, 0, 2]), tf.transpose(a=full_states, perm=[1, 0, 2])

    def get_weights(self):
        """ Get weights used in the network. 

        Allows for rebuilding or tweaking different weights to do experiments / analyses.

        Returns:
            dict: Dictionary of rnn weights including the following keys:

            :Dictionary Keys: 
                * **init_state** (*ndarray(dtype=float, shape=(1, :attr:`N_rec` *))*) -- Initial state of the network's recurrent units.
                * **W_in** (*ndarray(dtype=float, shape=(:attr:`N_rec`. :attr:`N_in` *))*) -- Input weights.
                * **W_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec` *))*) -- Recurrent weights.
                * **W_out** (*ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec` *))*) -- Output weights.
                * **b_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, *))*) -- Recurrent bias.
                * **b_out** (*ndarray(dtype=float, shape=(:attr:`N_out`, *))*) -- Output bias.
                * **Dale_rec** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- Diagonal matrix with ones and negative ones on the diagonal. If :data:`dale_ratio` is not ``None``, indicates whether a recurrent unit is excitatory(1) or inhibitory(-1).
                * **Dale_out** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- Diagonal matrix with ones and zeroes on the diagonal. If :data:`dale_ratio` is not ``None``, indicates whether a recurrent unit is excitatory(1) or inhibitory(0). Inhibitory neurons do not contribute to the output.
                * **input_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_in`*))*) -- Connectivity mask for the input layer. 1 where connected, 0 where unconnected.
                * **rec_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_rec`, :attr:`N_rec`*))*) -- Connectivity mask for the recurrent layer. 1 where connected, 0 where unconnected.
                * **output_connectivity** (*ndarray(dtype=float, shape=(:attr:`N_out`, :attr:`N_rec`*))*) -- Connectivity mask for the output layer. 1 where connected, 0 where unconnected.
                * **dale_ratio** (*float*) -- Dale's ratio, used to construct Dale_rec and Dale_out. Either ``None`` if dale's law was not applied, or 0 <= dale_ratio <=1 if dale_ratio was applied.

            Note:
                Keys returned may be different / include other keys depending on the implementation of :class:`RNN` used. A different set of keys will be included e.g. if the :class:`~psychrnn.backend.models.lstm.LSTM` implementation is used. The set of keys above is accurate and meaningful for the :class:`~psychrnn.backend.models.basic.Basic` and :class:`~psychrnn.backend.models.basic.BasicScan` implementations.
        """
        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True
      
        weights_dict = dict()
        
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name):
            # avoid saving duplicates
            if var.name.endswith(':0') and var.name.startswith(self.name):
                name = var.name[len(self.name)+1:-2]
                weights_dict.update({name: var.eval(session=self.sess)})
        weights_dict.update({'W_rec1_1': self.get_effective_W_rec1_1().eval(session=self.sess)})
        weights_dict.update({'W_in1_1': self.get_effective_W_in1_1().eval(session=self.sess)})
        weights_dict.update({'W_rec1_2': self.get_effective_W_rec1_2().eval(session=self.sess)})
        weights_dict.update({'W_in1_2': self.get_effective_W_in1_2().eval(session=self.sess)})

        weights_dict.update({'W_out': self.get_effective_W_out().eval(session=self.sess)})
        weights_dict['dale_ratio'] = self.dale_ratio
        return weights_dict

    def save(self, save_path):
        """ Save the weights returned by :func:`get_weights` to :data:`save_path`

        Arguments:
            save_path (str): Path for where to save the network weights.

        """

        weights_dict = self.get_weights()

        np.savez(save_path, **weights_dict)

        return

    def train(self, trial_batch_generator, train_params={}):
        """ Train the network.

        Arguments:
            trial_batch_generator (:class:`~psychrnn.tasks.task.Task` object or *Generator[tuple, None, None]*): the task to train on, or the task to train on's batch_generator. If a task is passed in, task.:func:`batch_generator` () will be called to get the generator for the task to train on.
            train_params (dict, optional): Dictionary of training parameters containing the following possible keys:

                :Dictionary Keys: 
                    * **learning_rate** (*float, optional*) -- Sets learning rate if use default optimizer Default: .001
                    * **training_iters** (*int, optional*) -- Number of iterations to train for Default: 50000.
                    * **loss_epoch** (*int, optional*) -- Compute and record loss every 'loss_epoch' epochs. Default: 10.
                    * **verbosity** (*bool, optional*) -- If true, prints information as training progresses. Default: True.
                    * **save_weights_path** (*str, optional*) -- Where to save the model after training. Default: None
                    * **save_training_weights_epoch** (*int, optional*) -- Save training weights every 'save_training_weights_epoch' epochs. Weights only actually saved if :data:`training_weights_path` is set. Default: 100.
                    * **training_weights_path** (*str, optional*) -- What directory to save training weights into as training progresses. Default: None.               
                    * **curriculum** (`~psychrnn.backend.curriculum.Curriculum` *object, optional*) -- Curriculum to train on. If a curriculum object is provided, it overrides the trial_batch_generator argument. Default: None.
                    * **optimizer** (`tf.compat.v1.train.Optimizer <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Optimizer>`_ *object, optional*) -- What optimizer to use to compute gradients. Default: `tf.train.AdamOptimizer <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer>`_ (learning_rate=:data:`train_params`['learning_rate']` ).
                    * **clip_grads** (*bool, optional*) -- If true, clip gradients by norm 1. Default: True
                    * **fixed_weights** (*dict, optional*) -- By default all weights are allowed to train unless :data:`fixed_weights` or :data:`W_rec_train`, :data:`W_in_train`, or :data:`W_out_train` are set. Default: None. Dictionary of weights to fix (not allow to train) with the following optional keys:

                        Fixed Weights Dictionary Keys (in case of :class:`~psychrnn.backend.models.basic.Basic` and :class:`~psychrnn.backend.models.basic.BasicScan` implementations)
                            * **W_in** (*ndarray(dtype=bool, shape=(:attr:`N_rec`. :attr:`N_in` *)), optional*) -- True for input weights that should be fixed during training.
                            * **W_rec** (*ndarray(dtype=bool, shape=(:attr:`N_rec`, :attr:`N_rec` *)), optional*) -- True for recurrent weights that should be fixed during training.
                            * **W_out** (*ndarray(dtype=bool, shape=(:attr:`N_out`, :attr:`N_rec` *)), optional*) -- True for output weights that should be fixed during training.

                        :Note:
                            In general, any key in the dictionary output by :func:`get_weights` can have a key in the fixed_weights matrix, however fixed_weights will only meaningfully apply to trainable matrices.

                    * **performance_cutoff** (*float*) -- If :data:`performance_measure` is not ``None``, training stops as soon as performance_measure surpases the performance_cutoff. Default: None.
                    * **performance_measure** (*function*) -- Function to calculate the performance of the network using custom criteria. Default: None.

                        :Arguments:
                            * **trial_batch** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimuli for :attr:`N_batch` trials.
                            * **trial_y** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Target output for the network on :attr:`N_batch` trials given the :data:`trial_batch`.
                            * **output_mask** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output mask for :attr:`N_batch` trials. True when the network should aim to match the target output, False when the target output can be ignored.
                            * **output** (*ndarray(dtype=bool, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Output to compute the accuracy of. ``output`` as returned by :func:`psychrnn.backend.rnn.RNN.test`.
                            * **epoch** (*int*): Current training epoch (e.g. perhaps the performance_measure is calculated differently early on vs late in training)
                            * **losses** (*list of float*): List of losses from the beginning of training until the current epoch.
                            * **verbosity** (*bool*): Passed in from :data:`train_params`.

                        :Returns:
                            *float* 

                            Performance, greater when the performance is better.
        Returns:
            tuple:
            * **losses** (*list of float*) -- List of losses, computed every :data:`loss_epoch` epochs during training.
            * **training_time** (*float*) -- Time spent training.
            * **initialization_time** (*float*) -- Time spent initializing the network and preparing to train.

        """
        if not self.is_built:
            self.build()

        t0 = time()
        # --------------------------------------------------
        # Extract params
        # --------------------------------------------------
        learning_rate = train_params.get('learning_rate', .001)
        training_iters = train_params.get('training_iters', 50000)
        loss_epoch = train_params.get('loss_epoch', 10)
        verbosity = train_params.get('verbosity', True)
        save_weights_path = train_params.get('save_weights_path', None)
        save_training_weights_epoch = train_params.get('save_training_weights_epoch', 100)
        training_weights_path = train_params.get('training_weights_path', None)
        curriculum = train_params.get('curriculum', None)
        optimizer = train_params.get('optimizer',
                                     tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate))
        clip_grads = train_params.get('clip_grads', True)
        fixed_weights = train_params.get('fixed_weights', None) # array of zeroes and ones. One indicates to pin and not train that weight.
        performance_cutoff = train_params.get('performance_cutoff', None)
        performance_measure = train_params.get('performance_measure', None)
        performance_counter_max = train_params.get('performance_counter_max', 3)

        if (performance_cutoff is not None and performance_measure is None) or (performance_cutoff is None and performance_measure is not None):
                raise UserWarning("training will not be cutoff based on performance. Make sure both performance_measure and performance_cutoff are defined")

        if curriculum is not None:
            trial_batch_generator = curriculum.get_generator_function()

        if not isgenerator(trial_batch_generator):
            trial_batch_generator = trial_batch_generator.batch_generator()

        # --------------------------------------------------
        # Make weights folder if it doesn't already exist.
        # --------------------------------------------------
        if save_weights_path != None:
            if path.dirname(save_weights_path) != "" and not path.exists(path.dirname(save_weights_path)):
                makedirs(path.dirname(save_weights_path))

        # --------------------------------------------------
        # Make train weights folder if it doesn't already exist.
        # --------------------------------------------------
        if training_weights_path != None:
            if path.dirname(training_weights_path) != "" and not path.exists(path.dirname(training_weights_path)):
                makedirs(path.dirname(training_weights_path))

        # --------------------------------------------------
        # Compute gradients
        # --------------------------------------------------
        grads = optimizer.compute_gradients(self.reg_loss)

        # --------------------------------------------------
        # Fixed Weights
        # --------------------------------------------------
        if fixed_weights is not None:
            for i in range(len(grads)):
                (grad, var) = grads[i]
                name = var.name[len(self.name)+1:-2]
#                 print("Variable name: ", name)
                if name in fixed_weights.keys():
                    grad = tf.multiply(grad, (1-fixed_weights[name]))
                    grads[i] = (grad, var)


        # --------------------------------------------------
        # Clip gradients
        # --------------------------------------------------
        if clip_grads:
            grads = [(tf.clip_by_norm(grad, 1.0), var)
                     if grad is not None else (grad, var)
                     for grad, var in grads]

        # --------------------------------------------------
        # Call the optimizer and initialize variables
        # --------------------------------------------------
        optimize = optimizer.apply_gradients(grads)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.is_initialized = True

        # --------------------------------------------------
        # Record training time for performance benchmarks
        # --------------------------------------------------
        t1 = time()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        epoch = 1
        batch_size = next(trial_batch_generator)[0].shape[0]
        losses = []
        if performance_cutoff is not None:
            performance = performance_cutoff - 1
        performance_list = [0]
        while epoch * batch_size < training_iters and (performance_cutoff is None or np.mean(performance_list) < performance_cutoff):

            batch_x, batch_y, output_mask, _ = next(trial_batch_generator)
            self.sess.run(optimize, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            # --------------------------------------------------
            # Output batch loss
            # --------------------------------------------------
            if epoch % loss_epoch == 0:
                reg_loss = self.sess.run(self.reg_loss,
                                feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                losses.append(reg_loss)
                if verbosity:
                    print("Iter " + str(epoch * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(reg_loss))

            # --------------------------------------------------
            # Allow for curriculum learning
            # --------------------------------------------------
            if curriculum is not None and epoch % curriculum.metric_epoch == 0:
                trial_batch, trial_y, output_mask, _ = next(trial_batch_generator)
                output, _ = self.test(trial_batch)
                if curriculum.metric_test(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
                    if curriculum.stop_training:
                        break
                    trial_batch_generator = curriculum.get_generator_function()

            # --------------------------------------------------
            # Save intermediary weights
            # --------------------------------------------------
            if epoch % save_training_weights_epoch == 0:
                if training_weights_path is not None:
                    self.save(training_weights_path + str(epoch))
                    if verbosity:
                        print("Training weights saved in file: %s" % training_weights_path + str(epoch))
            
            # ---------------------------------------------------
            # Update performance value if necessary
            # ---------------------------------------------------
            if performance_measure is not None:
                trial_batch, trial_y, output_mask, _ = next(trial_batch_generator)
                output, _ = self.test(trial_batch)
                performance = performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity)
                if verbosity:
                    print("performance: " + str(performance))
                performance_list.append(performance)
                if len(performance_list) > performance_counter_max:
                    performance_list.pop(0)
            epoch += 1

        t2 = time()
        if verbosity:
            print("Optimization finished!")
            print("Performance list: ", performance_list)
            print("Mean Performance: ", np.mean(performance_list))

        # --------------------------------------------------
        # Save final weights
        # ---------------sa-----------------------------------
        if save_weights_path is not None:
            self.save(save_weights_path+str(epoch))
            if verbosity:
                print("Model saved in file: %s" % save_weights_path +str(epoch))

        # --------------------------------------------------
        # Return losses, training time, initialization time
        # --------------------------------------------------
        return losses, (t2 - t1), (t1 - t0)

    def train_curric(self, train_params):
        """Wrapper function for training with curriculum to streamline curriculum learning.

        Arguments: 
            train_params (dict, optional): See :func:`train` for details.

        Returns:
            tuple: See :func:`train` for details.
        """
        # --------------------------------------------------
        # Wrapper function for training with curriculum
        # to streamline curriculum learning
        # --------------------------------------------------

        curriculum = train_params.get('curriculum', None)
        if curriculum is None:
            raise UserWarning("train_curric requires a curriculum. Please pass in a curriculum or use train instead.")
        
        losses, training_time, initialization_time = self.train(curriculum.get_generator_function(), train_params)

        return losses, training_time, initialization_time

    def test(self, trial_batch):
        """ Test the network on a certain task input.

        Arguments:
            trial_batch ((*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*): Task stimulus to run the network on. Stimulus from :func:`psychrnn.tasks.task.Task.get_trial_batch`, or from next(:func:`psychrnn.tasks.task.Task.batch_generator` ).
        
        Returns:
            tuple:
            * **outputs** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_out` *))*) -- Output time series of the network for each trial in the batch.
            * **states** (*ndarray(dtype=float, shape =(*:attr:`N_batch`, :attr:`N_steps`, :attr:`N_rec` *))*) -- Activity of recurrent units during each trial.
        """
        if not self.is_built:
            self.build()

        if not self.is_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.is_initialized = True

        # --------------------------------------------------
        # Run the forward pass on trial_batch
        # --------------------------------------------------
        outputs, states = self.sess.run([self.predictions, self.states],
                                        feed_dict={self.x: trial_batch})

#         print(states.shape)
        return outputs, states#1, states2
