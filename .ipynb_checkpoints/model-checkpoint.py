import torch
from torch.utils.data import Dataset
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from deepforest import main
from deepforest import utilities
from utils import *
from evaluation import *


class Model(main.deepforest):

    def __init__(self, num_classes=1, label_dict={"Tree": 0}, transforms=None):
        super().__init__(num_classes, label_dict, transforms)
        self.loggertb = TensorBoardLogger("tb_logs", name="tree_canopy")
        self.best_map = 0

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # allow for empty data if data augmentation is generated
        images, targets = batch

        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        return losses

    def training_epoch_end(self, outputs):
        """
        This method logs metrics and 15 best and 15 worst images to tensorboard on the epoch end
        """

        # Each epoch save current learning rate
        self.loggertb.log_metrics({
            'losses_train': sum([out['loss'] for out in outputs]) / len(outputs),
            'lr': self.lr_schedulers().get_last_lr()[0]}, step=self.current_epoch)
        self.loggertb.save()

        # For every 5 epoch run validation to log metrics
        if self.current_epoch % 5 == 0:
            results, f1, mAP = evaluate(self, csv=self.config["val"]["csv_file"], folder=self.config['val']['root_dir'])
            self.loggertb.log_metrics({'f1': f1, 'mAP': mAP}, step=self.current_epoch)
            if self.best_map < mAP:
                self.best_map = mAP
                torch.save(self.model, f'best_model_{self.current_epoch}.pt')
            self.model.train()
            self.loggertb.save()

    def evaluate(self,
                 csv_file,
                 root_dir,
                 iou_threshold=None, ):
        """Compute intersection-over-union and precision/recall for a given iou_threshold
        Args:
            csv_file: location of a csv file with columns "name","xmin","ymin","xmax","ymax","label", each box in a row
            root_dir: location of files in the dataframe 'name' column.
            iou_threshold: float [0,1] intersection-over-union union between annotation and prediction to be scored true positive
            savedir: optional path dir to save evaluation images
        Returns:
            results: dict of ("results", "precision", "recall") for a given threshold
        """
        # Load on GPU is available

        self.model.eval()
        self.model.score_thresh = self.config["score_thresh"]
        self.model.to(self.current_device)

        predictions = predict_file(model=self.model,
                                   csv_file=csv_file,
                                   root_dir=root_dir,
                                   device=self.current_device,
                                   iou_threshold=0.1)

        ground_df = pd.read_csv(csv_file)
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        # if no arg for iou_threshold, set as config
        if iou_threshold is None:
            iou_threshold = self.config["validation"]["iou_threshold"]
        results = evaluate2(predictions=predictions,
                            ground_df=ground_df,
                            root_dir=root_dir,
                            iou_threshold=iou_threshold, )

        # replace classes if not NUll, wrap in try catch if no predictions
        if not results["results"].empty:
            results["results"]["predicted_label"] = results["results"]["predicted_label"].apply \
                (lambda x: self.numeric_to_label_dict[x] if not pd.isnull(x) else x)
            results["results"]["true_label"] = results["results"]["true_label"].apply \
                (lambda x: self.numeric_to_label_dict[x])

        return results

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.config["train"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=50)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def load_dataset(self,
                     csv_file,
                     root_dir=None,
                     augment=False,
                     shuffle=True,
                     batch_size=1):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.
        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            augment: Whether to create a training dataset, this activates data augmentations
        Returns:
            ds: a pytorch dataset
        """
        ds = TreeDataset(csv_file=csv_file,
                         root_dir=root_dir,
                         transforms=self.transforms(augment=augment),
                         label_dict=self.label_dict)

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=utilities.collate_fn,
            num_workers=self.config["workers"],
        )

        return data_loader

    def create_trainer(self, logger=None, callbacks=[], **kwargs):
        """Create a pytorch lightning training by reading config files
        Args:
            callbacks (list): a list of pytorch-lightning callback classes
        """

        self.trainer = pl.Trainer(logger=logger,
                                  max_epochs=self.config["train"]["epochs"],
                                  gpus=self.config["gpus"],
                                  checkpoint_callback=False,
                                  accelerator=self.config["distributed_backend"],
                                  fast_dev_run=self.config["train"]["fast_dev_run"],
                                  callbacks=callbacks,
                                  **kwargs)

    def predict_image(self, image=None, path=None, return_plot=False):

        self.model.eval()
        self.model.score_thresh = self.config["score_thresh"]

        # Check if GPU is available and pass image to gpu
        result = predict_image(model=self.model,
                               image=image,
                               return_plot=return_plot,
                               device=self.current_device,
                               iou_threshold=self.config["nms_thresh"])

        # Set labels to character from numeric if returning boxes df
        if not return_plot:
            if not result is None:
                result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])

        return result

    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False,
                     use_soft_nms=False,
                     sigma=0.5,
                     thresh=0.001):

        self.model.eval()
        self.model.score_thresh = self.config["score_thresh"]
        self.model.nms_thresh = self.config["nms_thresh"]

        result = predict_tile(model=self.model,
                              raster_path=raster_path,
                              image=image,
                              patch_size=patch_size,
                              patch_overlap=patch_overlap,
                              iou_threshold=iou_threshold,
                              return_plot=return_plot,
                              use_soft_nms=use_soft_nms,
                              sigma=sigma,
                              thresh=thresh,
                              device=self.current_device)

        # edge case, if no boxes predictioned return None
        if result is None:
            print("No predictions made, returning None")
            return None

        # Set labels to character from numeric if returning boxes df
        if not return_plot:
            result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])

        return result
