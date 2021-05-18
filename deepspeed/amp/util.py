
def comm_table(type, volume, sb, nsb):
    """
        type: collective operation types: "allreduce" or "allgather"
        volume(int): communication volumne
        sb (int): slowest bandwidth
        nsb (int): number of node that contains GPU with sb
    """

    if type == "allreduce":
        return 2 * (nsb - 1)* volume / (nsb * sb)
    elif type == "allgather":
        raise
    else:
        raise

"""
Reverse mapping rank_map. For example, rank_map: {node1: [0,2], node2: [1,3]}
results in {0 : node1, 2 : node1, 1 : node2, 3 : node2}
"""
def rank2node(rank_map):
    ret = dict()
    for k, v in rank_map.items():
        for i in v:
            ret[i] = k
    return ret


