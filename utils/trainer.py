import os
import time

import detectron2.utils.events
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import cv2
import numpy as np
from utils.lr_scheduler import DelayedCosineAnnealingLR
from utils.save_best_model import BestCheckpointer
from utils.validation import LossEvalHook, ImageEvalHook, FullImageEvalHook

augmentations = [
    # T.Resize(512),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    # T.RandomRotation(angle=[-90, 90], expand=True, center=None, sample_style='range'),
    T.RandomExtent(scale_range=(0.7, 1.3), shift_range=(0.7, 1.3)),

    T.RandomBrightness(0.8, 1.2),
    T.RandomContrast(0.8, 1.2),
    T.RandomSaturation(0.9, 1),
]


class CustomTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True, augmentations=[]),
                
            )
        ))
        hooks.insert(-1, BestCheckpointer(
            self.cfg.TEST.EVAL_PERIOD,
            DetectionCheckpointer(
                model=self.model,
                save_dir=self.cfg.OUTPUT_DIR,
            )
        ))
        hooks.insert(-1, ImageEvalHook(
            trainer=self,
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            dataset_name=self.cfg.DATASETS.TEST[0]
        ))
        hooks.insert(-1, FullImageEvalHook(
            model=self.model,
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            current_eval=self.iter,
            sample_size=self.cfg.PATCH,
            input_dirs=['test_inp'],
            output_dir=os.path.join(self.cfg.OUTPUT_DIR, "test_image")
        ))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name=dataset_name,
            tasks=('segm',),
            distributed=True,
            output_dir=output_folder
        )

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(
            is_train=True,
            augmentations=augmentations if cfg.AUGMENT else [],
            image_format='RGB',
            use_instance_mask=True
        ))

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        # cosine annealing lr scheduler options
        scheduler_param = {
            "max_iters": cfg.SOLVER.MAX_ITER // 7,
            "delay_iters": 20,
            "eta_min_lr": 0,
            "optimizer": optimizer,
            # warmup options
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "warmup_method": cfg.SOLVER.WARMUP_METHOD,
            # multi-step lr scheduler options
            "milestones": cfg.SOLVER.STEPS,
            "gamma": cfg.SOLVER.GAMMA,
        }
        return DelayedCosineAnnealingLR(**scheduler_param)
    
    # def run_step(self):
    #     data = next(self._trainer._data_loader_iter)[0]
    #     print(data['instances'])
    #     img = data['image'].numpy().transpose(1, 2, 0)
    #     img = np.zeros((1024,1024,3))
    #     for v in data['instances'].gt_masks:
    #         l = []
    #         i=list(v[0])
    #         c = np.random.randint(0,255)
    #         color=(c,c,c)
    #         print(i)
    #         for j in range(0,len(i)-2,2):
    #             l.append((int(i[j]),int(i[j+1])))
    #             print(l)
    #             cv2.circle(img, l[-1], 10, color, 3)
    #
    #         print(l)
    #         print(i)
    #     print(data['image'].numpy().transpose(1, 2, 0).shape)
    #     cv2.imwrite("/media/deadman445/disk/test.png", data['image'].numpy().transpose(1, 2, 0))
    #     cv2.imwrite("/media/deadman445/disk/test_circle.png", img)
    #     exit(0)
    #     self._trainer.iter = self.iter
    #     self._trainer.run_step()
