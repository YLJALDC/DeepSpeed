import torch
from parallelism import parallelism
from abc import ABC
from util import comm_table, rank2node

class AmpModel(ABC):
    def __init__(self, model_config):
        self.model_config = model_config
        return

    @abstractmethod
    def estimate(self, parallelism: Parallelism, bs, alpha, beta, gamma):
        return

class gpt2(AmpModel):
    def __init__(self, model_config):
        super().init(model_config)
        self.h = model_config["hidden_size"]
        self.s = model_config["sequence_length"]
        self.n = model_config["num_layers"]
        self.v = model_config["vocab_size"]
        
      
        """
        PP note:
          Deepspeed GPT2 model layers pattern:
              0: EmbeddingPipe
              1: lambda
              2~(num_layer+1): ParallelTransformerLayerPipe
              num_layer+2: lambda
              num_layer+3: FusedLayerNorm
              num_layer+4: EmbeddingPipe
              num_layer+5: fp16_to_fp32

        Only count embed, transformer_layer, other as noop.
        """
        
        self._layer = ["embed2h", "noop"]
        for i in range(self.n):
            self._layer.append("transformer_layer")
        
        self._layer.extend(["noop","noop", "embed2v", "noop"])


    """Computation and Communication note:

        model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())

        A matrix multiplication of size A_{mxn}B_{nxk} involves 2mnk floating-point operations.

        p: number of mp 
        h: hidden size
        B: batch size
        S: sequence length
        V: vocabulary size
        n: number of layer
        n_: number of attention heads (actually no affect on the final results)

        EmbeddingPipe: 
                (1)VocabParallelEmbedding
                   - (B, S) * (v/p, h) -> (B, S, h) : 2BShv/p

                   - forward allreduce: BSh

                   - param count: h * v / p

                (2)position_embeddings
                   - 2Bsh(max_position_embedding)/p
                     max_position_embedding = 1024, v=50256. Linear in s,h. Can be ignored.

        ParallelTransformerLayerPipe:
                 (1)ParallelSelfAttention
                    (i) query_key_value transform: 
                        - (S, B, h) * (h, 3h/p)-> (S, B, 3 * n_/p * h/n_): 2BS * (3h^2) / p = 6BSH^2 / p
                        -backward allreduce (*): 3h^2 / p
                        -param count: 3 h**2 / p
                    (ii) QK scores: 
                        - (n_/p)  (B, S * h/n_) * (B, S * h/n_) -> (n_/p, B, S * S): 2(n_/p)BS^2(h/n_) = 2BS^2h / p

                    (iii) context:
                        - (n_/p) (B, S * S) * (B, S * h/n_) -> (n_/p, B, S * h/n_) = 2B(n_/p)S^2(h/n_) = 2BS^2h/p

                    (iv) dense:
                        - (S, B, h/p) * (h/p, h) -> (S, B, h) = 2SBh^2/p
                        - forward allreduce: BSh
                        - param count: h**2 / p

                 (2)ParallelMLP
                     (i) h to 4h:
                         - (S, B, h) * (h, 4h/p) -> (S, B, 4h / p): 2BS * (4h^2) / p = 8BSh^2 / p
                         - backward allreduce: 4h^2 / p
                         - param count: 4h**2 / p

                     (ii) 4h to h:
                         - (S, B, 4h / p) * (4h / p, h) -> (S, B, h) = SB (4h^2) / p = 8BSh^2 / p
                         - forward allreduce: BSh
                         - param count: 4h**2 / p

        https://github.com/microsoft/DeepSpeedExamples/blob/bdf8e59aede8c8e0577e8d4d557298ca8515268f/Megatron-LM-v1.1.5-3D_parallelism/megatron/mpu/cross_entropy.py#L28
        !!!_VocabParallelCrossEntropy(forward allreduce):
            (1) max logits: BS (ignore)
            (2) all logits: BSv
            (3) sum exp logits: BS (ignore)

        -> One layer of transformer:
           Comp: 24BSh^2 / p + 4BS^2h/p
           Comm: 7h^2 / p + 2BSh
           param: 12h**2/p

        EmbeddingPipe: 
                - (S, B, h) * (h, v/p) -> (S, B, v/p) -> (S, B, v/p) = 2BShv/p
                - backward allreduce: vh/p
                - param: h*v/p

       -> Total (counting backward):
          comp: 12nBSh/p * (6h + S) + 12BShv / p 
          comm: n*(7h^2/p + 2BSh) + BSh + vh/p


    """
    
    def estimate(self, parallelism: Parallelism, bs, alpha, beta, gamma):
        
        # cluster should be set at the entry point of amp.
        assert cluster is not None
        cluster_info = cluster.get_info()

        rank_map, pp, dp, mp, parts = parallelism.get_repr()
        
        # a reverse map that maps rank to node
        rank_node_map = rank2node(rank_map)

        h = self.model_config["hidden_size"]
        s = self.model_config["sequence_length"]
        n = self.model_config["num_layers"]
        v = self.model_config["vocab_size"]
        
        # build layer activation lookup table. Noop exatly has the same activation as the previous op.
        # Leave bs factor outside.
        layer_volume = []
        last_volume = 0
        for i in range(len(self._layer)):
            layer_type = self._layer[i]
            if layer_type == "embed2h" or layer_type == "transformer_layer":
                last_volume = s * h
                layer_volume.append(last_volume)
            elif layer_type == "embed2v":
                last_volume = s * v / mp
                layer_volume.append(last_volume)
            elif layer_type == "noop":
                layer_volume.append(last_volume)
            else:
                raise RuntimeError("Unknown layer type.")
        
        # this stores the mp+pp time for each data parallel replica 
        mp_pp_time = [0] * dp
        max_mp_pp_time = 0

        for i in range(dp):
            # This loops analysis the runtime for a single dp replica
            cur_mp_pp = 0
            for j in range(pp):
                
                # TODO: Hack deepspeed to accept a "parts" instead of
                # trying to search it itself.

                # key: node name, value: bandwidth
                bandwidth_dict = {}

                for k in range(mp):
                    # get the rank and information of current GPU.
                    cur_rank = parallelism.axis2rank(j, i, k)
                    cur_node = rank_node_map[cur_rank]
                    cur_bandwidth = cluster_info[cur_node]["bandwidth"] 
                    bandwidth_dict[cur_node] = cur_bandwidth

                for layer_id in range(parts[j], parts[j+1]):
                    layer_type = self._layer[layer_id]
                    if layer_type == "embed2h":
                        cur_mp_pp += 2 * bs * s * h * v / mp
                        cur_mp_pp += alpha * comm_table("allreduce", bs * s * h, bandwidth_dict)
                    if layer_type == "embed2v":
                        cur_mp_pp += 2 * bs * s * h * v / mp
                        cur_mp_pp += alpha * comm_table("allreduce", v * h / mp, bandwidth_dict)
                    elif layer_type == "transformer_layer":
                        cur_mp_pp += (24 * bs * s * h ** 2 + 4 * bs * s ** 2 * h) / mp
                        cur_mp_pp += alpha * comm_table("allreduce", 7*h**2/mp + 2*bs*s*h, bandwidth_dict)
                    elif layer_type == "noop":
                        pass
                    else:
                        raise RuntimeError("Unknown layer type.")
                
                # add loss calculation
                if j == (pp-1):
                    cur_mp_pp += alpha * comm_table("allreduce", bs * s * v, bandwidth_dict)
                
                # plus the time for communication between pipeline.
                # !!! If not last stage.
                if j != (pp -1):
                    slowest_bandwidth = float('inf')
                    for k in range(mp):
                        cur_rank = parallelism.axis2rank(j, i, k)
                        peer_rank = parallelism.axis2rank(j+1, i, k)
                        cur_node = rank_node_map[cur_rank]
                        peer_node = rank_node_map[peer_rank]
                    
                        # assuming intra-node does not take time.
                        if cur_node != peer_node:
                            cur_bandwidth = min(cluster_info[cur_node]["bandwidth"], cluster_info[peer_node]["bandwidth"])
                            if cur_bandwidth < slowest_bandwidth:
                                slowest_bandwidth = cur_bandwidth

                    # for example, if split is :
                    # stage 0: 
                    #     layer0-layer13(inclusive)
                    # stage 1:
                    #     layer14-layer29(inclusive)
                    # parts will be : [0, 14, 30]   
                    cur_mp_pp += beta * bs * (pp+1)  * layer_volume[parts[j+1]-1] / (pp * slowest_bandwidth)

            mp_pp_time[i] = cur_mp_pp
            if cur_mp_pp > max_mp_pp_time:
                max_mp_pp_time = cur_mp_pp
        
        # count dp gradient synchronization time
        max_dp_time = 0
        for i in range(pp):
            param_count = 0
            for layer_id in range(parts[i], parts[i+1):
                layer_type = self._layer[layer_id]
                if layer_type == "embed2h" or layer_type == "embed2v":
                    param_count += h * v / mp
                elif layer_type == "transformer_layer":
                    param_count += 12 * h ** 2 / mp
                elif layer_type == "noop":
                    pass
                else:
                    raise RuntimeError("Unknown layer type.")

            for j in range(mp):
                bandwidth_dict = {}
                for k in range(dp):
                    cur_rank = parallelism.axis2rank(i, k, j)
                    cur_node = rank_node_map[cur_rank]
                    cur_bandwidth = cluster_info[cur_node]["bandwidth"] 
                    bandwidth_dict[cur_node] = cur_bandwidth
                
                # note: number of param is the same, but we have different bandwidths
                cur_dp_time = comm_table("allreduce", param_count, bandwidth_dict)
                # local aggregation and update
                cur_dp_time += param_count * bs * gamma
                if cur_dp_time > max_dp_time:
                    max_dp_time = cur_dp_time
        
        max_dp_time *= beta

        return max_mp_pp_time + max_dp_time
