# Code inspired by U-Sleep article
# and https://github.com/neergaard/utime-pytorch

import torch
from csdp_training.lightning_models.base import Base_Lightning
from csdp_training.utility import log_test_step
from ml_architectures.usleep.usleep import USleep
import pytorch_lightning as pl
import os

class USleep_Lightning(Base_Lightning):
    def __init__(
        self,
        lr,
        batch_size,
        initial_filters = 5,
        complexity_factor = 1.67,
        progression_factor = 2,
        lr_patience = 50,
        lr_factor = 0.5,
        lr_minimum = 0.0000001,
        loss_weights = None,
        include_eog = True,
    ):
        num_channels = 2 if include_eog == True else 1

        inner = USleep(num_channels=num_channels,
                       initial_filters=initial_filters,
                       complexity_factor=complexity_factor,
                       progression_factor=progression_factor)
        
        super().__init__(inner,
                         lr, 
                         batch_size,
                         lr_patience,
                         lr_factor,
                         lr_minimum,
                         loss_weights)

        self.initial_filters = initial_filters
        self.complexity_factor = complexity_factor
        self.progression_factor = progression_factor
        self.include_eog = include_eog
        self.num_channels = num_channels

    def channels_prediction_EEGONLY(self, x_eegs, ybatch, tags):
        eegshape = x_eegs.shape
        
        num_eegs = eegshape[1]

        all_preds = {}

        for i in range(num_eegs):
            x_eeg = x_eegs[:,i,...]

            x_eeg = torch.unsqueeze(x_eeg, 1)

            pred = self(x_eeg)
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.squeeze(pred)
            pred = pred.swapaxes(0,1)
            pred = pred.to("cpu")

            eeg_tag = tags["eeg"][i]

            all_preds[f"{eeg_tag}"] = pred
                
        log_test_step("results",
                      f"{self.output_folder_prefix}", 
                      dataset=tags["dataset"],
                      subject=tags["subject"],
                      record=tags["record"],
                      preds=all_preds,
                      labels=ybatch.to("cpu"))
    
    def channels_prediction(self, x_eegs, x_eogs, ybatch, tags):
        eegshape = x_eegs.shape
        eogshape = x_eogs.shape
        
        num_eegs = eegshape[1]
        num_eogs = eogshape[1]
        
        assert eegshape[2] == eogshape[2]

        all_preds = {}

        for i in range(num_eegs):
            for p in range(num_eogs):
 
                x_eeg = x_eegs[:,i,...]
                x_eog = x_eogs[:,p,...]

                x_eeg = torch.unsqueeze(x_eeg, 1)
                x_eog = torch.unsqueeze(x_eog, 1)
                
                x_temp = torch.cat([x_eeg, x_eog], dim=1)
                
                pred = self(x_temp)
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.squeeze(pred)
                pred = pred.swapaxes(0,1)
                pred = pred.to("cpu")

                eeg_tag = tags["eeg"][i]
                eog_tag = tags["eog"][p]

                all_preds[f"{eeg_tag}/{eog_tag}"] = pred
                
        log_test_step("results",
                      f"{self.output_folder_prefix}", 
                      dataset=tags["dataset"],
                      subject=tags["subject"],
                      record=tags["record"],
                      preds=all_preds,
                      labels=ybatch.to("cpu"))
    
    def prep_batch(self, x_eeg, x_eog):

        assert len(x_eeg.shape) == 3, "EEG shape must be on the form (batch_size, num_channels, data)"
        assert x_eeg.shape[1] == 1, "Only one EEG channel allowed"

        if self.include_eog == True:
            assert len(x_eog.shape) == 3, "EOG shape must be on the form (batch_size, num_channels, data)"
            assert x_eog.shape[1] == 1, "Only one EOG channel allowed"
            xbatch = torch.cat((x_eeg, x_eog), dim=1)
        else:
            xbatch = x_eeg

        return xbatch

    def training_step(self, batch: dict, _):
        x_eeg = batch["eeg"]
        x_eog = batch["eog"]
        ybatch = batch["labels"]

        xbatch = self.prep_batch(x_eeg, x_eog)

        pred = self(xbatch)
        
        step_loss, _, _, _ = self.compute_train_metrics(pred, ybatch)

        self.training_step_outputs.append(step_loss)

        return step_loss

    def validation_step(self, batch: dict, _):
        # Step per record
        x_eeg = batch["eeg"]
        x_eog = batch["eog"]
        ybatch = batch["labels"]

        xbatch = self.prep_batch(x_eeg, x_eog)
        
        pred = self(xbatch)
        
        step_loss, step_acc, step_kap, step_f1 = self.compute_train_metrics(pred, ybatch)

        assert (step_acc != None) and (step_kap != None) and (step_f1 != None)

        self.validation_step_loss.append(step_loss)
        self.validation_step_acc.append(step_acc)
        self.validation_step_kap.append(step_kap)
        self.validation_step_f1.append(step_f1)

        pred = torch.swapdims(pred, 1, 2)
        pred = torch.reshape(pred, (-1, 5))
        pred = torch.argmax(pred, dim=1)
        ybatch = torch.flatten(ybatch)
        
        self.validation_labels.append(ybatch)
        self.validation_preds.append(pred)

    def run_test(self, 
                 trainer: pl.Trainer,
                 loader,
                 output_folder_prefix,
                 load_best_model = True):
        self.eval()
        
        os.makedirs(output_folder_prefix)

        self.output_folder_prefix = output_folder_prefix

        if load_best_model == True:
            _ = trainer.test(self, loader, ckpt_path="best")
        else:
            _ = trainer.test(self, loader)

    def test_step(self, batch: dict, _):
        # Step per record
        x_eeg: torch.Tensor = batch["eeg"]
        x_eog: torch.Tensor = batch["eog"]
        ybatch: torch.Tensor = batch["labels"]
        tags: dict = batch["tag"]

        assert len(x_eeg.shape) == 3
        ybatch = torch.flatten(ybatch)
        
        if self.include_eog == True:
            assert len(x_eog.shape) == 3
            self.channels_prediction(x_eeg, x_eog, ybatch, tags)
        else:
            channels_pred: torch.Tensor = self.channels_prediction_EEGONLY(x_eeg, ybatch, tags)
