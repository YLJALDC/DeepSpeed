import torch
from parallelism import parallelism
from abc import ABC
from util import comm_table

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

    def estimate(self, parallelism: Parallelism, bs, alpha, beta, gamma):
        rank_map, dp, mp, pp = parallelism.get_repr()

        h = self.model_config["hidden_size"]
        s = self.model_config["sequence_length"]
        n = self.model_config["num_layers"]
        v = self.model_config["vocab_size"]
        
        # mp part

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

                    (2)position_embeddings
                       - 2Bsh(max_position_embedding)/p
                         max_position_embedding = 1024, v=50256. Can be ignored.

            ParallelTransformerLayerPipe:
                     (1)ParallelSelfAttention
                        (i) query_key_value transform: 
                            - (S, B, h) * (h, 3h/p)-> (S, B, 3 * n_/p * h/n_): 2BS * (3h^2) / p = 6BSH^2 / p
                            -backward allreduce (*): 3h^2 / p

                        (ii) QK scores: 
                            - (n_/p)  (B, S * h/n_) * (B, S * h/n_) -> (n_/p, B, S * S): 2(n_/p)BS^2(h/n_) = 2BS^2h / p

                        (iii) context:
                            - (n_/p) (B, S * S) * (B, S * h/n_) -> (n_/p, B, S * h/n_) = 2B(n_/p)S^2(h/n_) = 2BS^2h/p

                        (iv) dense:
                            - (S, B, h/p) * (h/p, h) -> (S, B, h) = 2SBh^2/p
                            - forward allreduce: BSh

                     (2)ParallelMLP
                         (i) h to 4h:
                             - (S, B, h) * (h, 4h/p) -> (S, B, 4h / p): 2BS * (4h^2) / p = 8BSh^2 / p
                             - backward allreduce: 4h^2 / p

                         (ii) 4h to h:
                             - (S, B, 4h / p) * (4h / p, h) -> (S, B, h) = SB (4h^2) / p = 8BSh^2 / p
                             - forward allreduce: BSh

            -> One layer of transformer:
               Comp: 24BSh^2 / p + 4BS^2h/p
               Comm: 7h^2 / p + 2BSh

            EmbeddingPipe: 
                    - (S, B, h) * (h, v/p) -> (S, B, v/p) -> (S, B, v/p) = 2BShv/p
                    - backward allreduce: vh/p
                    
           -> Total (counting backward):
              comp: 12nBSh/p * (6h + S) + 12BShv / p 
              comm: n*(7h^2/p + 2BSh) + BSh + vh/p

        """
        t_mp_comp = 12 * n * bs * s * h / mp * (6 * h + s) + 12 * bs * s * h * v / mp
        
        layer_comm_time = comm_table("allreduce", volume = (7 * h ** 2 / mp) + \
                     2 * comm_table("allreduce", volume = s * bs * h)

        t_mp_comm = n * layer_time + comm_table("allreduce", volume = (s * bs * h)) + \
                    comm_table("allreduce", voulme = (h * v / mp))

        t_mp = t_mp_comp + alpha * t_mp_comm

        return


