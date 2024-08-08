"""Example of using the emmv_scores function with a model from the PyCaret library."""

from emmv import emmv_scores


def run():
    """Run the example."""
    # Adapted from https://pycaret.org/setup/
    # Importing dataset
    from pycaret.datasets import get_data

    anomalies = get_data('anomaly')

    # Importing module and initializing setup
    from pycaret.anomaly import *

    anomaly_setup = setup(data=anomalies)

    # create a model
    model = create_model('iforest')
    results = assign_model(model)

    # Get EM & MV scores
    test_scores = emmv_scores(model, anomalies)
    print('Excess Mass score;', test_scores['em'])
    print('Mass Volume score:', test_scores['mv'])


if __name__ == "__main__":
    run()
