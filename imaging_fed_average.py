from dataclasses import dataclass
from enum import Enum
from functools import reduce
from logging import WARNING
import numpy as np
from typing import Dict, Optional

from imaging_utilities import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays, Parameters
from lr_imaging_mnist_local import LRImagingLocal



class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


def aggregate_fit(
        results: Dict[str, Dict],
) -> NDArrays:
    """Aggregate fit results using weighted average."""

    # Convert results
    weights_results = [
        (fit_res["params"], fit_res["n_obs"])
        for _, fit_res in results.items()
    ]
    ndarrays_aggregated = aggregate(weights_results)
    return ndarrays_aggregated
    # TODO: Endgoal should have also the metrics below is the og format
    # # Aggregate custom metrics if aggregation fn was provided
    # metrics_aggregated = {}
    # if self.fit_metrics_aggregation_fn:
    #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
    #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
    # elif server_round == 1:  # Only log this warning once
    #     log(WARNING, "No fit_metrics_aggregation_fn provided")
    #
    # return parameters_aggregated, metrics_aggregated


# def imaging_fed_average(evaluate_fn, fit_round):
def aggregate_evaluate(
    self,
    server_round: int,
    results,  #: List[Tuple[ClientProxy, EvaluateRes]],
    failures,  #: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
):  # -> Tuple[Optional[float], Dict[str, Scalar]]:
    """Aggregate evaluation losses using weighted average."""
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    # if not self.accept_failures and failures:
    #     return None, {}

    # Aggregate loss
    loss_aggregated = weighted_loss_avg(
        [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
    )

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.evaluate_metrics_aggregation_fn:
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No evaluate_metrics_aggregation_fn provided")

    return loss_aggregated, metrics_aggregated


# helper functions
def weighted_loss_avg(results):  #: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


# : List[Tuple[NDArrays, int]]) -> NDArrays
def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
