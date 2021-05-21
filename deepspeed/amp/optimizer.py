from .parallelism import parallelism
class optimizer():
    def __init__(self, model, cluster, budget):
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
        self._estimator = model.estimate
 
        # hard coded number of runs to get the optimal setting
        self._optimal_budget = 1000
          
        self._cluster = cluster

    """
        sample a valid 3d parallelism configuration
    """
    def _sample(self):
        # return a fixed one for debug
        pp = 2
        dp = 2
        mp = 2
        parts = [0, 14, 30]
        
        rank_map = {"h0.ray-dev8.BigLearning": [0],
                    "h1.ray-dev8.BigLearning": [1],
                    "h2.ray-dev8.BigLearning": [2],
                    "h3.ray-dev8.BigLearning": [3],
                    "h4.ray-dev8.BigLearning": [4],
                    "h5.ray-dev8.BigLearning": [5],
                    "h6.ray-dev8.BigLearning": [6],
                    "h7.ray-dev8.BigLearning": [7],
                     }

        return parallelism(pp, dp, mp, parts, rank_map)

    """
        optimize model parameter based on trials. 
        https://github.com/petuum/adaptdl/blob/2c957652f4b5ffdc9899fdb8e365cca09bedc71c/adaptdl/adaptdl/goodput.py#L194
    """
    def _fit(self):
        #for i in range(len(self.budget)):
        #    _throughput = self._estimator.estimate(self.alpha, self.beta, self.gamma)
        pass

    """
        simulate a run
    """
    def _simulate(self):
        pass
    
    """
        Optimize based on available trials.
    """
    def optimize(self):
        for i in range(self._budget):
            # Compute the throughput using current model parameter
            sample = self._sample()
            self._estimator(sample, self._cluster, 3, self._alpha, self._beta, self._gamma)

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
            throughput = self._estimator(sample, self._cluster, 3, self._alpha, self._beta, self._gamma)
            if throughput > best_throughput:
                best_throughput = throughput
                best_setting = sample
        return best_setting
