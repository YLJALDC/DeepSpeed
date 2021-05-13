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

            EmbeddingPipe: 
                       
                       (1)

                       (2)

            

        """
        t_mp_comp = (6 * bs * s * h/ mp) * ( n * (12 * h + 2 * s) + 2 * v)
        layer_time = comm_table("allreduce", volume = (3 * h ** 2 + bs * s * h)) + \
                     2 * comm_table("allreduce", volume = s * bs * h) + \
                     comm_table("allreduce", volume = (4 * h ** 2 + bs * s * h))

        t_mp_comm = self.n * layer_time + comm_table("allreduce", volume = (s * bs * h)) + \
                    comm_table("allreduce", voulme = (h * v / mp + bs * s * h))
                    comm_table("allgather", volume = bs * s * v / mp)


        t_mp = t_mp_comp + alpha * t_mp_comm

        pass


