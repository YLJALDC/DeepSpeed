from esitimator import estimator
from AmpModel import AmpModel
from cluster import cluster

class optimizer():
    def __init__(self, model: AmpModel, cluster: cluster, budget):
        # cost model parameter
        self._alpha = 1
        self._beta = 1
        self._gamma = 1

        # future parameter that counts overlap
        self._theta = 1

        # used to store (Parallelism, int (parallelism)) tuple information
        self._trials = dict()

        # maximum number of trials
        self._budget = budget
        
        # throughput estimator
        self._estimator = estimator()
 
        # hard coded number of runs to get the optimal setting
        self._optimal_budget = 1000
          
        # model, should be rerepresented by amp
        self.model = model

    """
        sample a valid 3d parallelism configuration
    """
    def _sample(self):
        pass

    """
        optimize model parameter based on trials
    """
    def _fit(self):
        for i in range(len(self.budget)):
            _throughput = self._estimator.estimate(self.alpha, self.beta, self.gamma)
        pass

    """
        simulate a run
    """
    def _simulate(self):
        pass
    
    """
        Optimize based on available trials.
    """
    def _optimize(self):
        for i in range(self.budget):
            # Compute the throughput using current model parameter
            sample = self._sample()
            estimator.set_sample(sample)

            # run trials
            throughput = self._simulate()
            self._trials.append((sample, throughput))
            self._fit()
    
    """
        The sampler/estimator should run very fast, we maybe can just set a hard number 
        to get the optimal setting
    """
    def get_optimal(self):
        best_setting = None
        best_throughput = 0
        for i in range(self._optimal_budget):
            sample = self._sample()
            throughput = self._estimator.estimate(sample, self._alpha, self._beta, self._gamma)
            if throughput > best_throughput:
                best_throughput = throughput
                best_setting = sample
        return best_setting
