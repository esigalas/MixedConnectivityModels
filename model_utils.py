from psychrnn.tasks.task import Task
import numpy as np

class TargetDistractor(Task):
    """
    Class of the Target with Distractor Task without Motor Preparation Input.
    """
    def __init__(self, N_inputs, N_outputs, dt, tau, T, N_batch, cue_start=100, cue_end=300, dis_start=700,
                 dis_end=900, decision_start=1300, sigma=0.20, distractor_strength=0.5):
        super(TargetDistractor, self).__init__(N_inputs, N_outputs, dt, tau, T, N_batch)
        self.cue_start = cue_start
        self.cue_end = cue_end
        self.dis_start = dis_start
        self.dis_end = dis_end
        self.sigma = sigma
        self.decision_start = decision_start
        self.distractor_strength = distractor_strength
        self.decision_start = decision_start
        # set the number of channels for input cues
        self.input_cue_size = int(self.N_in)

    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.

        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.

        Returns:
            dict: Dictionary of trial parameters.

        """

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()

        # init input cue, distractor and motor prep channels (size 8 each)
        # 
        cue = np.zeros(self.N_in)
        distractor = np.zeros(self.N_in)
        
        # pick a random input channel (first 8) as the stimulus
        cue_value = np.random.randint(self.input_cue_size) 
        # pick one of the rest 7 channels as a distractor
        distractor_value = cue_value
        while distractor_value == cue_value:
            distractor_value = np.random.randint(self.input_cue_size)
        
        cue[cue_value] = 1
        distractor[distractor_value] = self.distractor_strength
        
        params['cue'] = cue
        params['cue_start'] = self.cue_start
        params['cue_end'] = self.cue_end
        
        params['distractor'] = distractor
        params['distractor_start'] = self.dis_start
        params['distractor_end'] = self.dis_end
        
#         params['motor_prep'] = motor_prep

        params['sigma'] = self.sigma
        params['decision_start'] = self.decision_start
        return params

    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.

        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.

        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()

        Returns:
            tuple:

            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.

        """

        # ----------------------------------
        # Initialize
        # ----------------------------------
        x_t = np.zeros(self.N_in)
        y_t = np.zeros(self.N_out)
#         mask_t = np.ones(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        actual_cue = params['cue']
        cue_start = params['cue_start']
        cue_end = params['cue_end']
        
        actual_distractor = params['distractor']
        distractor_start = params['distractor_start']
        distractor_end = params['distractor_end']

        sigma = params['sigma']
        decision_start = params['decision_start']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if time < cue_start:
            x_t += np.zeros_like(actual_cue)#+ np.random.normal(scale=sigma, size=actual_cue.shape)
        if cue_start < time < cue_end:
            x_t += actual_cue + np.random.normal(scale=sigma, size=actual_cue.shape)
        if cue_end < time < distractor_start:
            x_t += np.zeros_like(actual_cue) #+ np.random.normal(scale=sigma, size=actual_cue.shape)
        if distractor_start < time < distractor_end:
            x_t += actual_distractor + np.random.normal(scale=sigma, size=actual_distractor.shape)
        if distractor_end < time < decision_start:
            x_t += np.zeros_like(actual_distractor)# + np.random.normal(scale=sigma, size=actual_distractor.shape)
        
        if decision_start < time:
            x_t += np.zeros_like(actual_cue)#+ np.random.normal(scale=sigma, size=actual_cue.shape)
            y_t = actual_cue[:self.input_cue_size]
            mask_t = np.ones_like(y_t)
        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """

        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))

def get_bump_attractor_weights(neurons, number_of_inputs, kappa=200, motor_preparation=True,
                               population_size=3/4, population_overlap=1/2, groupsize=10, negative_w=True,
                               ordered=False, bump_overlap=10, amplitude=1, bump_regulator=0.8):
    """
    Function that creates the bump attractor reccurent weights along with the bump input weights.
    *Imported implementation based on Matlab code.*
    Modified to include negative weights.
    
    :param neurons: (int) number of neurons
    :param number_of_cues: (int) the number of cues of the task
    :param kappa: (int) the kappa parameter that decides the width of the bump
    :param population_size: (float) the size of the neuron populations for spatial memory and motor preparation
    :param population_overlap: (float) the overlap between the two populations
    :param groupsize: (int) the number of neurons within a bump
    :return: 
            bump_weights: (numpy array) reccurent weights of the bump model
            input_weights: (numpy array) input weights of the bump model
            v: the 1D bump signal (mainly just for visualisation) 
    """
    # set the theta parameter basically a vector from 0 to 2π with step such it includes all neurons once
    theta = (np.arange(0, neurons) / neurons * 2 * np.pi)
    # if we do not want negative weights in the bump attractor model
    if not negative_w:
        v1 = np.exp(kappa * np.cos(theta))
        v = v1 / sum(v1)
    else:
        # calculate a second bump with bigger width but smaller magnitude
        v1 = np.exp(kappa * np.cos(theta))
        v2 = np.exp((kappa-(kappa*bump_regulator)) * np.cos(theta))
        # subtract the two bumps and multiply the result by 6 to get the desired bump model with negative weights
        v = ((v1 / sum(v1)) - (v2 / sum(v2)))
    
    # initialize the recurrent weights as zeros
    bump_weights = np.zeros((v.shape[0], v.shape[0]))
    # roll the bump across the 2D-array to finalize the recurrent weights of the bump model
    for i in range(len(bump_weights)):
        bump_weights[:, i] = np.roll(v, i)
    
    if motor_preparation:
        # initialize the bump input weights as zero
        input_weights = np.zeros((neurons, number_of_inputs))
        # initialize the weights for each bump group to zeros
        template = np.zeros((int(neurons* population_size)))
        # try non zero input weights for the neurons not in the bump
    #     input_weights = np.random.normal(0, 0.15,size=(neurons, number_of_inputs))
    #     template = np.random.normal(0, 0.15, size=(int(neurons* population_size)))
        # init offset 
        offset=0
        rstep = np.random.randint(100)
        rstep2 = np.random.randint(100)
        # mix the order of the input cue list according to a different random number for stim and motor prep bumps
        cue_list = np.concatenate([np.roll(np.arange(number_of_inputs)[:int(number_of_inputs/2)], rstep),
                                   np.roll(np.arange(number_of_inputs)[int(number_of_inputs/2):], rstep2)])
        # for each stimulus set the bump input weights
        for cue_input in cue_list: 
            # set the sinoid signal as the input weights for the bump group (multiplied by amplitude: default 1)
            template[:groupsize] = np.sin(np.arange(0, 0.999, 1 / groupsize) * np.pi) * amplitude

            # roll offset the bump randomly across the respective column of the input weight array
            if ordered:# and cue_input < int(number_of_inputs/2):
                offset += bump_overlap
            else:
                offset = np.random.randint(1, np.ceil(neurons* population_size))

            if cue_input < int(number_of_inputs/2):
                input_weights[:int(neurons*population_size),  cue_input] = np.roll(template, offset)
            else:
                input_weights[int(np.ceil(neurons*(population_size-population_overlap))):int(np.ceil(neurons*(2*population_size-population_overlap))), cue_input] = np.roll(template, offset)
    else:
        input_weights = np.zeros((neurons, number_of_inputs))
        template = np.zeros((int(neurons* population_size)))
        offset=0
        rstep = np.random.randint(100)
        cue_list = np.roll(np.arange(number_of_inputs)[:int(number_of_inputs)], rstep)
        for cue_input in cue_list:
            template[:groupsize] = np.sin(np.arange(0, 0.999, 1 / groupsize) * np.pi) * amplitude
            offset += bump_overlap
            input_weights[:int(neurons*population_size),  cue_input] = np.roll(template, offset)
        
    return bump_weights, input_weights, v

def performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
    """
    Function that calculates the performance of the model on the given trial batch and the output of the model
    *task should be initialized beforehand
    """
    return drt.accuracy_function(trial_y, output, output_mask)

def create_model_dataset(model, task, trials=500):
    """
    Function that creates a dataset of neuron responses given a model and a task.
    """
    # get two batches of trials from the specified task
    trial_inputs = []
    trial_gt_outputs = []
    trial_mask = []
    for i in np.arange(int(trials/task.N_batch)):
        x,y,m,_ = task.get_trial_batch()
        trial_inputs.append(x)
        trial_gt_outputs.append(y)
        trial_mask.append(m)
    
    # concatenate the ground truth outputs that will be used as label for each trial
    data_gt = np.concatenate([x for x in trial_gt_outputs])
    # concatenate the mask output ( to be used for calcualting accuracy in batch and correct trials)
    data_masks = np.concatenate([x for x in trial_mask])
    # iterate through trial inputs and pass them through the model to get the model output and model/neuron states
    outputs_list = []
    states_list = []
    for trial_input in trial_inputs:
        outputs, states = model.test(trial_input)
        outputs_list.append(outputs)
        states_list.append(states)
    model_outputs = np.concatenate([x for x in outputs_list])
    neuron_states = np.concatenate([x for x in states_list])
    # set a one-hot label vector that matches number of trials
    neuron_states_labels = data_gt[:,-1:,:].reshape(data_gt.shape[0],data_gt.shape[2])

    return neuron_states, neuron_states_labels, model_outputs, data_gt, data_masks

def get_correct_trials(correct_output, test_output, output_mask):
    """
    Function that picks only the correct trials of a model 
    """
    chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
    truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
    return np.equal(chosen, truth)
