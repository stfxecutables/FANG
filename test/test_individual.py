from pathlib import Path
from typing import Any, Union

import numpy as np

from src.exceptions import VanishingError
from src.individual import Individual

LOGFILE = Path(__file__).resolve().parent / "test_individual_outputs.txt"


def test_individual(capsys: Any) -> None:
    # SEED = 2
    # np.random.seed(SEED)
    # random.seed(SEED)

    MAX_ATTEMPTS = 10
    vanishings = 0
    descs = []
    torch_descs = []
    fitnesses = []
    errs = []

    def log(ind: Union[Individual, str]) -> None:
        if isinstance(ind, Individual):
            fitnesses.append(ind.fitness)
            descs.append(str(ind))
            torch_descs.append(str(ind.sequential_model))
        elif isinstance(ind, str):
            fitnesses.append(None)
            descs.append("Error")
            torch_descs.append("Error")
            errs.append(ind)
        else:
            raise ValueError()

    for i in range(MAX_ATTEMPTS):
        try:
            ind = Individual(
                n_nodes=10, task="classification", input_shape=(1, 28, 28), output_shape=10
            )
        except VanishingError:
            vanishings += 1
            fitnesses.append(0)
            continue
        except RuntimeError as e:
            if "mat1 dim 1 must match mat2 dim 0" in str(e):
                with capsys.disabled():
                    print(ind)
                    print(ind.sequential_model)
            raise e
        try:
            with capsys.disabled():
                print(ind)
                print(ind.sequential_model)
                ind.evaluate_fitness()
                log(ind)
        except RuntimeError as e:
            if "Output size is too small" in str(e):
                vanishings += 1
                log(str(e))
            elif "Kernel size can't be greater than actual input size" in str(e):
                vanishings += 1
                log(str(e))
            elif "Padding size should be less than the corresponding input dimension" in str(e):
                vanishings += 1
                log(str(e))
            elif "mat1 dim 1 must match mat2 dim 0" in str(e):
                with capsys.disabled():
                    print(ind)
                    print(ind.sequential_model)
                    log(str(e))
                raise e
            else:
                raise e

    with capsys.disabled():
        messages = [f"Number of networks where size shrunk to zero: {vanishings}\r\n"]
        messages.append("Fitnesses of networks after 1 epoch (accuracies):\r\n")
        for i, fitness in enumerate(fitnesses):
            messages.append(f"Model {i}: {str(float(np.round(fitness, 3)))}")
        for i, (desc, fitness) in enumerate(zip(descs, fitnesses)):
            messages.append("\r\n\r\n")
            messages.append("=" * 80)
            messages.append(f"Model {i}: {str(float(np.round(fitness, 3)))}")
            messages.append("=" * 80)
            messages.append(desc)

        messages.append("\r\nErrors:")
        for err in errs:
            messages.append(err)

        with open(LOGFILE, "w") as logfile:
            logfile.write("\r\n".join(messages))
        for message in messages:
            print(message)
