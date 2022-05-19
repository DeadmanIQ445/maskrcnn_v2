from pathlib import Path
from typing import List

import pandas as pd
import rtree
from shapely.geometry import Polygon
from shapely.wkb import dumps, loads


def get_smallest_polygon_area(json_file: Path):
    """Return pixel-wise are of polygon"""
    df = pd.read_json(json_file)

    df['area'] = df.polygon.map(lambda x: Polygon(x).area)

    return df['area'].min()


def instance_f1_score(gt: List[Polygon],
                      pred,
                      _format,
                      iou_threshold=0.5,
                      v: bool = True):
    """
    Measures instance f1-score for two sets of polygons.
    The algorithm description can be found on
    https://medium.com/the-downlinq/the-spacenet-metric-612183cc2ddb
    If the format = 'point' True Positive counts when the prediction point lies within the polygon of GT
    :param gt: list of shapely Polygons, represents ground truth;
    :param pred: list of shapely Polygons or Points (according to the 'format' param, represents prediction;
    :param _format: 'vector' or 'point', means format of prediction and corresponding variant of algorithm;
    :param iou_threshold: uoi threshold (0.5 by default)
    :param v: is_verbose
    :return: float, f1-score and string, log
    """
    log = ''
    gt_rtree_index = rtree.index.Index()

    # for some reason builtin pickling doesn't work
    for i, polygon in enumerate(gt):
        gt_rtree_index.insert(
            i, polygon.bounds, dumps(polygon)
        )

    if _format == 'vector':
        tp = sum(map(_has_match_rtree,
                     (dumps(polygon) for polygon in pred),
                     [iou_threshold] * len(pred),
                     [gt_rtree_index] * len(pred)))
    else:  # format = 'point'
        tp = sum(map(_lies_within_rtree,
                     (point for point in pred),
                     [gt_rtree_index] * len(pred)))
    fp = len(pred) - tp
    fn = len(gt) - tp
    # to avoid zero-division
    if tp == 0:
        f1 = 0.
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    if v:
        log += 'True Positive = ' + str(tp) + ', False Negative = ' + str(fn) + ', False Positive = ' + str(fp) + '\n'

    return f1, log


def _has_match_rtree(polygon_serialized, iou_threshold, groundtruth_rtree_index):
    """ Compares the polygon with the rtree index whether it has a matching indexed polygon,
    and deletes the match if it is found
    :param polygon_serialized: polygon to be matched
    :param iou_threshold: minimum IoU that is required for the polygon to be considered positive example
    :param groundtruth_rtree_index: rtree index
    :return: True if match found, False otherwise
    """
    polygon = loads(polygon_serialized)
    best_iou = 0
    best_item = None
    candidate_items = [
        candidate_item
        for candidate_item
        in groundtruth_rtree_index.intersection(
            polygon.bounds, objects=True
        )
    ]

    for item in candidate_items:
        candidate = loads(item.get_object(loads))
        metric = iou(polygon, candidate)
        if metric > best_iou:
            best_iou = metric
            best_item = item

    if best_iou > iou_threshold and best_item is not None:
        groundtruth_rtree_index.delete(best_item.id,
                                       best_item.bbox)
        return True
    else:
        return False


def _lies_within_rtree(point, groundtruth_rtree_index):
    """ Searches whether there is an indexed polygon which contains the point
    and deletes the match if it is found
    :param point: point to be matched
    :param groundtruth_rtree_index: rtree index
    :return: True if match found, False otherwise
    """
    candidate_items = [
        candidate_item
        for candidate_item
        in groundtruth_rtree_index.intersection(
            (point.x, point.y), objects=True
        )
    ]
    for item in candidate_items:
        candidate = loads(item.get_object(loads))
        if candidate.contains(point):
            groundtruth_rtree_index.delete(item.id, candidate.bounds)
            return True
    return False


def iou(polygon1: Polygon, polygon2: Polygon):
    # buffer(0) may be used to “tidy” a polygon
    # works good for self-intersections like zero-width details
    # http://toblerity.org/shapely/shapely.geometry.html
    poly1_fixed = polygon1.buffer(0)
    poly2_fixed = polygon2.buffer(0)
    return poly1_fixed.buffer(0).intersection(poly2_fixed).area / poly1_fixed.union(poly2_fixed).area
