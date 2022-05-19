import datetime
import logging
import random
import time
import copy
import os
from aeronet.dataset import BandCollection
from aeronet.dataset.utils import parse_directory

from typing import Union
import geopandas as gpd

from rasterio.transform import xy
import cv2
import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import Visualizer, ColorMode
from aeronet.dataset import Predictor
import pandas as pd
import shapely.topology
import slidingwindow
from shapely.geometry import Polygon
from tqdm import tqdm
import rasterio
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from rasterio.plot import reshape_as_image


class ImageEvalHook(HookBase):
    def __init__(self, trainer: DefaultTrainer, eval_period, dataset_name, seed=42, num=10, current_eval=0):

        self._trainer = trainer
        self._period = eval_period
        self._dataset = dataset_name
        self._seed = seed
        self._num = num
        self._current_eval = current_eval

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_image_logging()

    def _do_image_logging(self):
        random.seed(self._seed)

        dataset = DatasetCatalog.get(self._dataset)
        metadata = MetadataCatalog.get(self._dataset)
        predictor = DefaultPredictor(self._trainer.cfg)

        self._current_eval += 1
        os.makedirs(f'detect_test_image/{self._current_eval}', exist_ok=True)
        os.makedirs(f'gt_test_image/{self._current_eval}', exist_ok=True)
        os.makedirs(f'mask_test_image/{self._current_eval}', exist_ok=True)
        self._num = min(self._num, len(dataset))
        for ii, im_dict in enumerate(random.sample(dataset, self._num)):
            # im = cv2.imread(im_dict["file_name"])
            with rasterio.open(im_dict['file_name']) as f:
                im = reshape_as_image(f.read())
            outputs = predictor(im)

            v1 = Visualizer(im, scale=1, metadata=metadata)
            v2 = Visualizer(im, scale=1, metadata=metadata)

            v = Visualizer(
                im[:, :, ::-1],
                metadata=metadata,
                scale=0.8,
            )
            for box in outputs["instances"].pred_boxes.to('cpu'):
                v.draw_box(box)
                v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
            v = v.get_output()
            img = v.get_image()
            cv2.imwrite(f'detect_test_image/{self._current_eval}/{ii}.png', img)
            prediction: np.array = v1.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            reference: np.array = v2.draw_dataset_dict(im_dict).get_image()
            cv2.imwrite(f'mask_test_image/{self._current_eval}/{ii}.png', prediction)
            cv2.imwrite(f'gt_test_image/{self._current_eval}/{ii}.png', reference)

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class FullImageEvalHook(HookBase):
    def __init__(self,
                 eval_period,
                 model,
                 output_dir,
                 current_eval,
                 sample_size=(512,512),
                 channels='rgb',
                 overlap=0.5,
                 input_dirs=['test_inp'],
                 seed=42,
                 num=50):
        self._model = model
        self._period = eval_period
        self._seed = seed
        self._num = num
        self._sample_size = sample_size
        self._channels = channels
        self._overlap = overlap
        self._input_dirs = input_dirs
        self._output_dir = output_dir
        self._current_eval = current_eval
        self.device = torch.device('cuda')
        self.channels = ('RED', 'GRN', 'BLU')

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_image_logging()

    def _do_image_logging(self):
        random.seed(self._seed)
        input_channels = self._compose_bands(self._input_dirs)
        band_collection = BandCollection(input_channels)
        predictor = InstanceSegmentationBatchPredictor(
            sample_size=self._sample_size,
            input_channels=self._channels,
            output_labels="prediction_reference",
            processing_fn=self,
            batch_size=1,
            bound=0,
            dtype=np.float32,
            overlap=self._overlap)
        instances = predictor.process(
            bc=band_collection,
            output_directory=os.path.dirname("prediction_reference"))

        def proc_row(row):
            polygon = np.asarray(row, dtype=np.float_)
            polygon[:, 0], polygon[:, 1] = xy(band_collection.transform, row[:, 1], row[:, 0])
            return shapely.geometry.Polygon(polygon)
        if instances.empty:
            print('empty')
        else:
            instances['geometry'] = instances['polygon'].apply(proc_row)
            del instances['polygon']
            out_path = os.path.join(self._output_dir,
                                    str(datetime.datetime.now().strftime("%H:%M:%S")
                                        + '_' + str(self._current_eval)))
            self._current_eval+=1
            os.makedirs(out_path, exist_ok=True)
            gp = gpd.GeoDataFrame(instances, geometry='geometry')
            gp = gp.set_crs(band_collection.crs)
            gp.to_file(str(out_path))
            gp.to_file(self._output_dir+'.geojson', driver='GeoJSON')

    def __call__(self, x):
        x, coords = preprocess(x.transpose(1, 2, 0))
        with torch.no_grad():
            self._model.eval()
            height, width = x.shape[:2]
            image = torch.as_tensor(x.astype("float32").transpose(2, 0, 1), device=self.device)

            inputs = {"image": image, "height": height, "width": width}
            y = self._model([inputs])
            y = y[0]['instances'].to('cpu')
            self._model.train()
        masks, scores, classes = postprocess(y, coords)
        # masks = y.pred_masks.numpy().astype("uint8")
        # scores = y.scores.numpy()
        # classes = y.pred_classes.numpy()

        df = pd.DataFrame(columns=['polygon', 'class', 'score'])

        for ii in range(len(masks)):
            mask = np.expand_dims(masks[ii], axis=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                polygon = max(contours, key=cv2.contourArea).reshape(-1, 2)

                df = df.append({
                    "polygon": np.array(polygon),
                    "class": classes[ii],
                    "score": scores[ii]
                }, ignore_index=True)

        return df

    def _compose_bands(self, bands_dirs: Union[os.PathLike, str]):
        bands = []

        if type(bands_dirs) == str:
            bands_dirs = [bands_dirs]

        for bands_dir in bands_dirs:
            for channel in self.channels:
                band = parse_directory(bands_dir, [channel])
                assert len(band) == 1, f"channel '{channel}' not found in '{bands_dir}'"
                bands.extend(band)

        if len(self.channels)*len(bands_dirs) != len(bands):
            raise Exception(f'''Couldn't parse all required bands from directories provided, 
                                list of successfully parsed bands: {bands}''')
        return bands

def _iou_(test_poly, truth_poly):
    """Intersection over union"""
    try:
        test_poly, truth_poly = Polygon(test_poly), Polygon(truth_poly)
        intersection_result = test_poly.intersection(truth_poly)
        intersection_area = intersection_result.area
        union_area = test_poly.union(truth_poly).area
        if union_area == 0:
            return 0, 0, 0
        iou = intersection_area / union_area
        if test_poly.area==0 or truth_poly.area==0:
            return  0, 0 ,0
        iotest = intersection_area / test_poly.area
        iotruth = intersection_area / truth_poly.area
        return iou, iotest, iotruth


    except shapely.topology.TopologicalError:
        return 0, 0, 0


def join_nms(mosaic_df: pd.DataFrame, iou_threshold, corr_coef):
    ret_boxes = []
    boxes_cp = copy.deepcopy(mosaic_df.to_dict('records'))

    while len(boxes_cp) > 0:
        m = min(range(len(boxes_cp)), key=lambda i: boxes_cp[i]['area'])

        b_m = boxes_cp[m]
        boxes_cp.pop(m)
        flag = True
        for i in boxes_cp:
            poly_m = b_m['polygon']
            poly_i = i['polygon']
            iou, poly_m_inter, poly_i_inter = _iou_(poly_m, poly_i)
            if poly_m_inter > corr_coef:
                flag = False
                break
            if poly_i_inter > corr_coef:
                continue

            if iou > iou_threshold:
                if b_m['score'] < i['score']:
                    flag = False
                    break
        if flag:
            ret_boxes.append(b_m)
    return ret_boxes

def preprocess(x: np.array):
    """Filling image with zeros to make image square"""
    orig_height, orig_width = x.shape[:2]

    new_dim = max(orig_height, orig_width)
    new_image = np.zeros(shape=(new_dim, new_dim, 3), dtype=np.uint8)

    new_image[:orig_height, :orig_width] = x[:orig_height, :orig_width]
    return new_image, (orig_height, orig_width)


def postprocess(instance, coords) :
    """
    Cropping image, filled with zeros

    :return masks and scores
    """
    height, width = coords

    cropped_masks = []

    for ii, mask in enumerate(instance.pred_masks.numpy().astype("uint8")):
        cropped_masks.append(mask[:height, :width])

    cropped_masks = np.array(cropped_masks)
    scores = instance.scores.numpy()
    pred_classes = instance.pred_classes.numpy()

    return cropped_masks, scores, pred_classes


class InstanceSegmentationBatchPredictor(Predictor):

    def __init__(self,
                 input_channels,
                 output_labels,
                 processing_fn,
                 batch_size=16,
                 sample_size=(512, 512),
                 bound=128,
                 n_workers=1,
                 verbose=True,
                 overlap=0.5,
                 **kwargs):

        # sample_size = (9698, 9606)

        super().__init__(input_channels, output_labels, processing_fn,
                         sample_size, bound, n_workers, verbose, **kwargs)
        self.batch_size = batch_size
        self.overlap = overlap

    def compute_windows(self, numpy_image, patch_size, patch_overlap):
        if patch_overlap > 1:
            raise ValueError("Patch overlap {} must be between 0 - 1".format(patch_overlap))
        # Generate overlapping sliding windows
        windows = slidingwindow.generate(numpy_image,
                                         slidingwindow.DimOrder.HeightWidthChannel,
                                         patch_size, patch_overlap)
        return windows


    def process(self, bc, output_directory):
        # Compute sliding window index
        image = bc.numpy().transpose(1, 2, 0)
        windows = self.compute_windows(image, self.sample_size[0], self.overlap)
        # Save images to tempdir
        predicted_instances = []
        for index, window in enumerate(tqdm(windows)):
            # crop window and predict
            crop = image[windows[index].indices()]

            # crop is RGB channel order, change to BGR?
            instances = self.processing_fn(crop.transpose(2, 0, 1))
            if instances is not None:
                # transform the coordinates to original system
                xmin, ymin, xmax, ymax = windows[index].getRect()

                for ii in range(len(instances)):
                    instances.loc[ii].polygon[:, 0] += xmin
                    instances.loc[ii].polygon[:, 1] += ymin

                predicted_instances.append(instances)
        if len(predicted_instances) == 0:
            print("No predictions made, returning None")
            return None

        predicted_instances = pd.concat(predicted_instances)

        # Confidence interval
        # predicted_instances = predicted_instances[predicted_instances.score > 0.5]

        # Non-max supression for overlapping boxes among window
        if self.overlap == 0:
            mosaic_df = predicted_instances
        else:
            print(
                f"{predicted_instances.shape[0]} predictions in overlapping windows, applying non-max supression"
            )
            mosaic_df = predicted_instances
            mosaic_df = mosaic_df[mosaic_df['polygon'].apply(len)>2]
            mosaic_df['area'] = mosaic_df.polygon.map(lambda x: Polygon(x).area)
            mosaic_df = mosaic_df[mosaic_df['area']>0]
            mosaic_df = join_nms(mosaic_df, iou_threshold=0.75, corr_coef=0.75)
            print(f"{len(mosaic_df)} predictions kept after non-max suppression")
            mosaic_df = pd.DataFrame(mosaic_df)
            # mosaic_df = mosaic_df[mosaic_df['score']>0.5]
            mosaic_df = mosaic_df.reset_index()

        return mosaic_df