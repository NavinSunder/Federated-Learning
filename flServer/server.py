import flwr as fl
import numpy as np

def aggregate_metrics(metrics):
    total_samples = 0
    weighted_accuracy = 0.0
    weighted_loss = 0.0

    print(f"[Server] Received Raw Metrics: {metrics}")  # Debugging Print

    for num_samples, client_metrics in metrics:
        if "accuracy" in client_metrics and "loss" in client_metrics:
            weighted_accuracy += num_samples * float(client_metrics["accuracy"])
            weighted_loss += num_samples * float(client_metrics["loss"])
            total_samples += num_samples


    mean_accuracy = (weighted_accuracy / total_samples) if total_samples > 0 else 0.0
    mean_loss = (weighted_loss / total_samples) if total_samples > 0 else 1.0

    print(f"[Server] Aggregated -> Accuracy: {mean_accuracy:.4f}, Loss: {mean_loss:.4f}")

    return {"accuracy": mean_accuracy, "loss": mean_loss}


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=aggregate_metrics
)


config = fl.server.ServerConfig(num_rounds=10)


fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=config,
    strategy=strategy
)
