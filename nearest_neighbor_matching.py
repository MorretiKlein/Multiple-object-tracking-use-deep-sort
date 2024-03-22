import numpy as np

def cosine_distance(a, b ,data_is_normalized = True):
    """
    compute pair-wise consine distance 
    a: matrix of N sample
    b : matrix of M sample 
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis = 1, keepdims= True)
        b = np.asarray(b) / np.linalg.norm(b, axis= 1, keepdims= True)
    return 1 - a @b.T
def nn_cosine_distance(a, b):
    distance = cosine_distance(a,b, data_is_normalized= False)
    return distance.min(axis = 0)

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget = None):
        # budget : Optional[int]: if not None, limit of sample, if the budget exceeded, the oldest models will be discarded 
        #Optional[Int] allows us to create an object which may or may not contain a int value.
        if metric == "cosine":
            self.metric = nn_cosine_distance
        else:
            raise ValueError("invalid metric")
        
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_target):
        """
        Update the distance metric with new data
        Store feature of new object and update sample list
        arg:
        features (ndarray): Array of features of new objects (N samples, M dimensions).
        targets (ndarray): Array containing IDs of objects corresponding to features.
        active_targets (List[int]): List of active (present) objects in the current frame.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target,[]).append(feature) # key is target and value is feature
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:] # keep budget the last element of the list
        self.samples = {k: self.samples[k] for k in active_target}

    def distance(self,features, targets):
        """compute distance between feature and target"""
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self.metric(self.samples[target], features)
        return cost_matrix