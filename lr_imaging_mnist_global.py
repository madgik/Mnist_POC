from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from imaging_fed_average import aggregate_fit
import imaging_utilities as utils


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        if server_round > 0:
            server_round -= 1
        return loss, {"accuracy": accuracy}, server_round

    return evaluate


# Start Flower server for five rounds of federated learning
class LRImagingGlobal:
    def __init__(self):
        self.model = LogisticRegression()

    def calculate_aggregates(self, fit_res_all: Dict):
        parameters_aggregated = aggregate_fit(
            fit_res_all
        ) if fit_res_all else None

        utils.set_model_params(self.model, parameters_aggregated)

        aggregated_model = self.model.fit(parameters_aggregated)
        model_params = utils.get_model_parameters(aggregated_model)
        return aggregate_fit(model_params)

    def check_evaluation(self):
        # check self.eval_result in order to proceed to next round and return result
        pass
