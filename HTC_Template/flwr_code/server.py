from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
# strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
#                   min_available_clients=1)

strategy = FedAvg(
    # With 1 client, 10% of 1 => 0; so set fractions to 1.0 to always request the available client
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    # And lower the minimums to 1 so sampling never demands 2 clients
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    # Optional: don't fail the round if a client drops mid-round
    accept_failures=True,
    evaluate_metrics_aggregation_fn=weighted_average,
)
# Define config
config = ServerConfig(num_rounds=20)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:5006",
        config=config,
        strategy=strategy,
    )