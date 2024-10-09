#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import logging
import statistics
import sys

from tqdm import tqdm

from metrics import best_subspan_em
from utils import read_file

logger = logging.getLogger(__name__)

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]


def main(input_path):
    all_examples = read_file(input_path)

    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []

    if 'kv' in input_path:
        for example in tqdm(all_examples):
            model_answer = example["model_answer"]
            accuracy = 1.0 if example["golden"].lower() in model_answer.lower() else 0.0
            all_example_metrics.append(({"accuracy": accuracy}, example))

        # Average metrics across examples
        for metric_name in ["accuracy"]:
            average_metric_value = statistics.mean(
                example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
            )
            logger.info(f"{metric_name}: {average_metric_value}")

    elif 'qa' in input_path:
        for example in tqdm(all_examples):
            all_example_metrics.append(get_metrics_for_example(example))

        # Average metrics across examples
        for (_, metric_name) in METRICS:
            average_metric_value = statistics.mean(
                example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
            )
            logger.info(f"{metric_name}: {average_metric_value}")


def get_metrics_for_example(example):
    gold_answers = example["golden"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with model predictions and answers.", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(args.input_path)
    logger.info("finished running %s", sys.argv[0])
