
def mp_table(type, volume, sb, nsb):
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

def infer_batch_size(config):
    pass

def infer_sync_points(config):
    pass
