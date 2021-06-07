import numpy as np
from .parallelism import parallelism
from .util import factor

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

class MCMCOptimizer():
    def __init__(self, model, cluster, budget):
        # cost model parameter
        self._alpha = 1
        self._beta = 1
        self._gamma = 0

        self._MCMC_BETA = 1

        # future parameter that counts overlap
        self._theta = 1

        # maximum number of trials
        self._budget = budget
        
        self._model = model
        
        # throughput estimator
        self._estimator = model.estimate

        # hard coded number of runs to get the optimal setting
        self._optimal_budget = 1000
          
        self._cluster = cluster
        self._world_size = len(self._cluster.get_info().keys())

    """
        propose a valid 3d parallelism configuration
    """
    def _propose(self):
        factors = factor(self._world_size)
        
        pp = np.random.choice(factors)
        factors = factor(self._world_size // pp)

        dp = np.random.choice(factors)
        mp = self._world_size // (pp * dp)
        
        
        # keep parts for uniform split and rank_map constant for now
        parts = self._model.uniform_split(pp)

        rank_map = {}
        rank = 0
        for key, value in self._cluster.get_info().items():
            rank_map[key] = [rank]
            rank += 1

        return parallelism(pp, dp, mp, parts, rank_map)
    
    """
        Optimize based on available trials.
    """
    def optimize(self):
        estimate_time = dict()

        cur = self._propose()
        cur_t = self._estimator(cur, self._cluster, 8, self._alpha, self._beta, self._gamma)
       
        best = cur
        best_t = cur_t

        estimate_time[cur] = cur_t
        
        for i in range(self._budget):
            # get a new proposal
            next = self._propose()
            next_t = self._estimator(next, self._cluster, 8, self._alpha, self._beta, self._gamma)
            acc_p = min(1, np.exp(self._MCMC_BETA * (cur_t - next_t)))

            rand = np.random.uniform(0, 1)
            if rand < acc_p:
                cur = next
                cur_t = next_t

            if cur_t < best_t:
                best_t = cur_t
                best = cur
            
            _, pp, dp, mp, _ = next.get_repr()
            
            if next not in estimate_time.keys():
                estimate_time[(pp,dp,mp)] = next_t
            else:
                assert next_t == estimate_time[(pp,dp,mp)], "estimator not deterministic"

        sorted_estimation = dict(sorted(estimate_time.items(), key=lambda item: item[1]))
        print(sorted_estimation)
        return best
