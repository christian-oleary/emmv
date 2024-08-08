"""Example of using the emmv_scores function with a model from the PyCaret library."""

from pycaret.anomaly import assign_model, create_model, setup
from pycaret.datasets import get_data

from emmv import emmv_scores


def run():
    """Run the example."""
    # Adapted from https://pycaret.org/setup/
    anomalies = get_data('anomaly')
    setup(data=anomalies)

    # create a model
    model = create_model('iforest')
    assign_model(model)

    # Get EM & MV scores
    scores = emmv_scores(model, anomalies)
    print(f'Excess Mass: {scores[0]}\nMass Volume: {scores[1]}')


if __name__ == "__main__":
    run()
