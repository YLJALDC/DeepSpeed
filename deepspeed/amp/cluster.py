import yaml
from collections import defaultdict

class cluster():
    def __init__(self, resource_file):
        self.resource_file = resource_file
        self.cluster_info = defaultdict(dict)
        self._parse_resource()

    def _parse_resource(self):
        resource_info = yaml.safe_load(open(self.resource_file, 'r'))
        for node in resource_info.pop('nodes', {}):
            node_dict = self.cluster_info[node.get("address")]
            node_dict["gpus"] = node.get("gpus")
            node_dict["bandwidth"] = node.get("bandwidth")
    
    def get_info(self):
        return self.cluster_info
