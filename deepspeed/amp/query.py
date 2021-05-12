import argparse
#from analysis import get_bert_time

def parse_args():

    parser = argparse.ArgumentParser(description='Runtime analysis.')
    parser.add_argument('--name', type=str, help='model name, currently supports ["bert"]')
    parser.add_argument('--pp', type=int, help='pipeline parallelism size')
    parser.add_argument('--mp', type=int, help='model parallelism size')
    parser.add_argument('--num_layers', type=int, help='number of layers for transformer')
    parser.add_argument('--hidden_size', type=int, help='hidden size for transformer')
    parser.add_argument('--num_attn_heads', type=int, help='number of attention heads for transformer')
    parser.add_argument('--batch_size', type=int, help='batch size')

    return parser.parse_args()

"""
Receive the configuration of a cluster and model, and return:
    (1) The rank of each GPU
    (2) The placement group lists
"""

def query_amp_topo(cluster):
    
    test_ranks = [2,0,3,5,7,1,4,6]
    ret = dict()
    count = 0
    for k, v in cluster.items():
        ret[k] = [test_ranks[count]]
        count += 1
    
    return ret, 4, 2

    # args = parse_args()
    # name = args.name
    # pp = args.pp
    # mp = args.mp
    # num_layers = args.num_layers
    # hidden_size = args.hidden_size
    # num_attn_heads = args.num_attn_heads
    # batch_size = args.batch_size

    # print(args)


    # if name == "bert":
    #    get_bert_time(pp=pp, mp=mp, num_layers=num_layers, hidden_size=hidden_size,
    #              num_attn_heads=num_attn_heads, batch_size=batch_size)

    # else:
    # raise ValueError("Not implemented model type.")


    # print(args.accumulate(args.integers))
