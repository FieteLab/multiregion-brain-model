import numpy as np
from scipy.ndimage import shift
import random
import torch

def step_func(v, threshold):
    """
    Step function: 1 if value >= threshod, 0 otherwise
    Return numpy array
    """
    
    return np.array(v >= threshold, dtype=int)

class SurprisalEngine():
    """
    Computes (binary) surprisal of sensory input based on previous inputs
    Handles multiple runs simultaneously

    Attributes
        nruns - int, number of runs the sensory inputs are from
    """
    
    def __init__(self, nruns):
        self.nruns = nruns

    # to be filled by subclasses
    def get_surprisal(self, s, return_raw=False):
        """
        Returns binary surprisal values of sensory inputs from each run

        Inputs:
            s - np.array, size (nruns, Ns), current sensory inputs
            return_raw - bool, whether to also return intermediate surprisal values
                before being converted to binary

        Outputs:
            if return_raw = True: surprisal
            if return_raw = False: (surprisal, raw_surprisal)
                surprisal - np.array, size (self.nruns, 1, 1), binary (0/1) surprisal values
                raw_surprisal - np.array, size (self.nruns, 1, 1), raw surprisal 
        """
        pass

class SurprisalLinear(SurprisalEngine): # using moving average of previous tau patterns
    """
    Computes surprisal by comparing current input with the average of previous *tau* observed inputs
    (takes the dot product between the two, compares the negative with *threshold*)
    Keeps track of recent inputs 

    Additional attributes:
        past_patterns - list[np.array], stores past tau inputs, 
            each element is size (nruns, Ns), has tau elements
        tau - int, number of past inputs to consider
        threshold - int/float, threshold for the negative of the dot product
            (if above threshold, surprisal = 1)
    """
    def __init__(self, tau, threshold, nruns, normalized=False, flipped=False):
        super().__init__(nruns)
        self.past_patterns = []
        self.tau = tau
        self.threshold = threshold
        self.normalized = normalized
        self.flipped = flipped

    def get_surprisal(self, s, return_raw=False):
        """
        If return_raw=True, returns binary surprisal 
            & raw surprisal (negative of the dot product)
        """
        if len(self.past_patterns) == 0: # if it's the first input, surprisal = 0
            self.past_patterns.append(s)
            surprisal = np.zeros((self.nruns, 1, 1))
            if return_raw:
                return surprisal, surprisal
            return surprisal
        
        # compute surprisal
        Wns = np.mean(self.past_patterns, axis=0)
        similarity = (Wns[:, None, :]@s[:, :, None]).squeeze(axis=(1, 2)) # dot product
        if self.normalized:
            similarity = similarity / (np.linalg.norm(Wns[:, None, :], axis=(1, 2))
                * np.linalg.norm(s[:,:, None], axis=(1, 2)))
            raw_surprisal = 1-similarity
        else:
            raw_surprisal = -similarity
        if self.flipped:
            raw_surprisal = -raw_surprisal
        surprisal = step_func(raw_surprisal, self.threshold)
        raw_surprisal = raw_surprisal[:, None, None]
        surprisal = surprisal[:, None, None]

        # update stored patterns
        if len(self.past_patterns) == self.tau:
            del self.past_patterns[0]
        self.past_patterns.append(s)

        if return_raw:
            return surprisal, raw_surprisal
        return surprisal
    
class SurprisalExponential(SurprisalEngine): # using decaying average of previous patterns
    """
    Computes surprisal by comparing current input with the exponential moving average of all previous inputs
    (takes the dot product between the two, compares the negative with *threshold*)
    (the average is iteratively computed as Wns = tau * Wns + (1-tau) * <new input>)

    Additional attributes:
        Wns - np.array, size (nruns, Ns), exponential moving average of previous inputs
        tau - float, weight on older inputs in computing the moving average
            (lower tau corresponds to faster decay of old inputs, larger weight on recent inputs)
        threshold - int/float, threshold for the negative of the dot product
            (if above threshold, surprisal = 1)
    """
    def __init__(self, tau, threshold, nruns, normalized=False, flipped=False):
        super().__init__(nruns)
        self.Wns = None
        self.tau = tau
        self.threshold = threshold
        self.nruns = nruns
        self.normalized = normalized
        self.flipped = flipped

    def get_surprisal(self, s, return_raw=False):
        """
        If return_raw=True, returns binary surprisal 
            & raw surprisal (negative of the dot product)
        """
        if self.Wns is None: # <=> it's the first input
            self.Wns = s
            surprisal = np.zeros((self.nruns, 1, 1))
            if return_raw:
                return surprisal, surprisal
            return surprisal
        
        # compute surprisal
        similarity = (self.Wns[:, None, :]@s[:, :, None]).squeeze(axis=(1, 2)) # dot product
        if self.normalized:
            similarity = similarity / (np.linalg.norm(self.Wns[:, None, :], axis=(1, 2))
                * np.linalg.norm(s[:,:, None], axis=(1, 2)))
            raw_surprisal = 1-similarity
        else:
            raw_surprisal = -similarity
        if self.flipped:
            raw_surprisal = -raw_surprisal
        surprisal = step_func(raw_surprisal, self.threshold)
        raw_surprisal = raw_surprisal[:, None, None]
        surprisal = surprisal[:, None, None]

        # update the exponential moving average
        self.Wns = self.tau * self.Wns + (1-self.tau) * s

        if return_raw:
            return surprisal, raw_surprisal
        return surprisal

class SurprisalExponentialBVC(SurprisalExponential): # using decaying average of previous patterns
    def __init__(self, b, tau, threshold, nruns, normalized=False, flipped=False):
        super().__init__(tau, threshold, nruns, normalized, flipped)
        self.b = b

    def get_surprisal(self, s, v=0, return_raw=False):
        if self.Wns is not None:
            Wns = self.Wns.reshape(self.nruns, self.b, self.b)
            Wns = shift(Wns, v)
            self.Wns = Wns.reshape(self.nruns, self.b*self.b)
        return super().get_surprisal(s, return_raw)

class SurprisalRegion():
    """
    Wrapper around SurprisalEngine to implement surprisal region --
    If an input has high surprisal above *threshold*, enters into special surprised state
    and only exits when input's surprisal is below *return_threshold*

    Attributes:
        surprisal_engine - SurprisalEngine, computes raw non-binary surprisal of input
        threshold - int/float, if surprisal exceeds this threshold then enters surprised state
        return_threshold - int/float
        high_surprisal - np.array(bool), size (nruns,), whether it is in surprised state
    """
    def __init__(self, surprisal_engine, threshold, return_threshold=None):
        """
        return_threshold: if None, will set to threshold
        """
        self.surprisal_engine = surprisal_engine
        self.threshold = threshold
        if return_threshold is None:
            return_threshold = threshold
        self.return_threshold = return_threshold
        self.high_surprisal = np.zeros(surprisal_engine.nruns, dtype=bool)
    def visit(self, s, update_state=True, **kwargs):
        """
        Inputs:
            s - np.array, size (nruns, Ns), current sensory inputs
            update_state - bool, whether to update current state
                (use this to ignore surprisal at the start of a path)

        Outputs: np.array, size (nruns,)
            end_current_map - np.array(bool), whether surprised state is entered
            start_new_map - np.array(bool), whether surprised state is exited
            raw_surprisal - np.array(float), raw non-binary surprisal 
        """
        surprisal_raw = self.surprisal_engine.get_surprisal(s, return_raw=True, **kwargs)[1].squeeze()
        # whether to enter surprised state
        end_current_map = np.logical_and(~self.high_surprisal, surprisal_raw > self.threshold)
        # whether to exit surprised state
        start_new_map = np.logical_and(self.high_surprisal, surprisal_raw <= self.return_threshold)
        if update_state:
            self.high_surprisal = np.logical_or(
                end_current_map, 
                np.logical_and(self.high_surprisal, surprisal_raw > self.return_threshold))
        return end_current_map, start_new_map, surprisal_raw

class SurprisalRegionCheat():
    def __init__(self, nruns, surprisal_enter_indices, surprisal_exit_indices):
        self.nruns = nruns
        self.surprisal_enter_indices = surprisal_enter_indices
        self.surprisal_exit_indices = surprisal_exit_indices
        self.high_surprisal = np.zeros(nruns, dtype=bool)
    def visit(self, x, update_state=True):
        end_current_map = np.zeros((self.nruns, 1, 1))
        start_new_map = np.zeros((self.nruns, 1, 1))
        if update_state:
            if x in self.surprisal_enter_indices:
                self.high_surprisal = ~self.high_surprisal
                end_current_map = np.ones((self.nruns, 1, 1))
            elif x in self.surprisal_exit_indices:
                self.high_surprisal = ~self.high_surprisal
                start_new_map = np.ones((self.nruns, 1, 1))
        surprisal_raw = self.high_surprisal.astype(int).reshape(-1, 1, 1)
        return end_current_map, start_new_map, surprisal_raw