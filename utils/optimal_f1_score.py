import logging
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
    Author: https://github.com/asharakeh/pod_compare/blob/7f16c406bb258a0de96e2492797f6e960f21f8ec/src/offline_evaluation/compute_average_precision.py
"""

from detectron2.data import MetadataCatalog

from utils.register_dataset import register_dataset

import click

logging.basicConfig(stream=sys.stdout, format="[F1 evaluator] %(message)s", level=logging.INFO)


@click.command(name="Calculated optimal f1 by trying different conf. scores thresholds")
@click.option('--dataset-path', required=True, type=Path, help="Path to dataset [only test will be evaluated]")
@click.option('--coco-file', required=True, type=Path, help="Path to coco_instances_results.json")
def calculate_optimal_f1(dataset_path: Path, coco_file: Path) -> None:
    assert dataset_path.is_dir()
    assert coco_file.is_file()

    logger = logging.getLogger()

    register_dataset(dataset_path)
    meta_catalog = MetadataCatalog.get("uchastok_test")

    # Evaluate detection results
    gt_coco_api = COCO(meta_catalog.json_file)
    res_coco_api = gt_coco_api.loadRes(str(coco_file))
    results_api = COCOeval(gt_coco_api, res_coco_api, iouType='segm')

    results_api.params.catIds = [1, 3]  # list(meta_catalog.thing_dataset_id_to_contiguous_id.keys())

    # Calculate and print aggregate results
    results_api.evaluate()
    results_api.accumulate()
    results_api.summarize()

    # Compute optimal micro F1 score threshold. We compute the f1 score for
    # every class and score threshold. We then compute the score threshold that
    # maximizes the F-1 score of every class. The final score threshold is the average
    # over all classes.
    precisions = results_api.eval['precision'].mean(0)[:, :, 0, 2]
    recalls = np.expand_dims(results_api.params.recThrs, 1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_f1_score = f1_scores.argmax(0)
    scores = results_api.eval['scores'].mean(0)[:, :, 0, 2]
    optimal_score_threshold = [scores[optimal_f1_score_i, i] for i, optimal_f1_score_i in enumerate(optimal_f1_score)]
    optimal_score_threshold = np.array(optimal_score_threshold)
    optimal_score_threshold = optimal_score_threshold[optimal_score_threshold != 0]
    optimal_score_threshold = optimal_score_threshold.mean()

    logger.info("Classification Score at Optimal F-1 Score: {}".format(optimal_score_threshold))

    text_file_name = coco_file.parent / 'results.txt'

    text_file_name.write_text(str(results_api.stats.tolist() + [optimal_score_threshold, ]))

    logger.info(f"Write results to {str(text_file_name)}")


if __name__ == '__main__':
    calculate_optimal_f1()
