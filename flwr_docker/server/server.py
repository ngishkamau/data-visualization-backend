#-*- coding:utf-8 -*-

from typing import List, Tuple

import os
import flwr as fl
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

class Client(fl.server.SimpleClientManager):
    def __init__(self) -> None:
        super().__init__()

    def register(self, client: ClientProxy) -> bool:
        addr = client.cid[5:]
        ip, _ = addr.split(':')
        with open(os.getcwd() + '/clients', 'a+') as f:
            f.write(ip + '\n')
        return super().register(client)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    client_manager=Client()
)