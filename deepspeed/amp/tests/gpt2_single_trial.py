from deepspeed.amp import query_amp_topo

orca_cluster = "../orca_resource.yml"
gpt2_model_config = {"name": "gpt2", "sequence_length": 1024, "hidden_size": 512, "num_layers": 24, "vocab_size": 50512}

ret = query_amp_topo(orca_cluster, gpt2_model_config, 1)



