"""
Stores information for each dimension of parallelism
"""
class parallelism():
    def __init__(self, bs: int, dp: int, mp: int, pp: tuple):

        self.bs = bs
        self.dp = dp
        self.mp = mp
        self.pp = pp

        # The order of axis is (dp, mp, pp). For example, if dp=2,mp=3,pp=4,
        # the numbe gpu at number 5 stands for (dp=0, mp=1, pp=1)

        self.rank_map = dict()
   
    """
    Should be set from optimizer. A valid rank map looks like:
    {node_1: [1,2], node_2: [3,0], ...}
    """
    def set_rank_map(self, rank_map):
        self._check_valid_rank(rank_map)
        self.rank_map = rank_map

    def _check_valid_rank(self, rank_map):
        base = []
        for k, v in rank_map.items():
            base.extend(v)

        validity = (sorted(base) == list(range(len(base))))
        bijection = (len(rank_map) == self.dp * self.mp * self.pp)
        assert validity && bijection, "rank map is not a permutation from 0 to range(num_gpus)."

    """
    Returns this as (rank_map, dp, mp, pp)
    """
    def get_repr(self):
        return (self.rank_map, self.dp, self.mp, self.pp)



