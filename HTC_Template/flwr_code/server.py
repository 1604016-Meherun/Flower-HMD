import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,   # set to 2+ when you add more PCs
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
