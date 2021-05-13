
"""
class for estimating the throughput
"""
class estimator():
    def __init__(self, model):
        self.model = model
        self.parallelism = None
        
    def estimate(self, alpha, beta, gamma):
        # do some computation here
        throughput = 0
        self.paralleism = None
        
        return throughput

    def set_parallelism(self, parallelism):
        self.parallelism = parallelism
