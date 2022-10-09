import argparse
import sys
import datetime
from logging import log
from torch import nn
from torch.utils.data import DataLoader
from dsets import LunaDataset
import math
from torch.optim import SGD
import numpy as np
from utils import enumerateWithEstimate



#this helps us use the command line
import torch.cuda


METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv =  sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,


        )

        parser.add_argument('--balanced',
help="Balance the training data to half positive, half negative.", action='store_true',
default=False,
)

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        #sets all the parameters, initialize the model, optimizer, and make sure
        #the CLI can call

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                #if the model has more than one GPU, we will use more.
                #send the model parameters to the gpu before constructing the optimizer.

            model = model.to(self.device)



    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
    #lr -0.001 and momentum 0.99 are safe choices.

    def initTrainDl(self):
        train_ds = LunaDataset(
        val_stride = 10,
                     isValSet_bool = False,
            ratio_int=int(self.cli_args.balanced),

        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
        train_ds,
        batch_size = batch_size, \
                     num_workers = self.cli_args.num_workers, \
                                   pin_memory = self.use_cuda,
        )

        return train_dl

    def main(self):
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)


    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        #initialization of an empty metrics array
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        #sets up the batch looping with time estimate
        batch_iter = enumerateWithEstimate(
            train_dl,
            'E{} Training'.format(epoch_ndx),
            start_ndx = train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var =  self.computeBatchLoss(batch_ndx,
                                              batch_tup,
                                              train_dl.batch_size,
                                              trnMetrics_g)

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    #trnMetrics_g tensor collects per-class metrics during training.
    #good for big projects





class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3,
                                padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, conv_channels, kernel_size=3,
                                padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2,2)

    def forward(self, input_batch):
        out = self.relu1(self.conv1(input_batch))
        out = self.relu2(self.conv2(out))
        return self.maxpool(out)





class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)
        self.head_linear = nn.Linear(1152,2)
        self.head_softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        conv_flat = block_out.view(block_out.size(0),-1)
        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_softmax(linear_output)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



    #computeBatchLoss function - called by both the training and validation loops
    #computes loss over a batch of samples

    #also feeds the batch into the model and computes the per-batch loss, using
    #Cross-Entropy loss

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        #take the inputs, run them through the model

        logits_g, probability_g = self.model(input_g)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g,
                           label_g[:,1],)



        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,

            )

        batch_iter = enumerateWithEstimate(
            val_dl,
            "E{} Validation ".format(epoch_ndx),
            start_ndx=val_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.computeBatchLoss(
                batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
        return valMetrics_g.to('cpu')


    #outputting performance metrics

    def logMetrics(self,
                   epoch_ndx,
                   mode_str, #whethr the metrics are training or validation
                   metrics_t,
                   classificationThreshold = 0.5):

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float32(truePos_count + falseNeg_count)


        metrics_dict['loss/all'] = \
            metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        metrics_dict['pr/f1_score'] = 2 * (precision*recall)/(precision+recall)

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            ))
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})").format(
          epoch_ndx,
          mode_str + '_neg',
          neg_correct=neg_correct,
          neg_count=neg_count,
          **metrics_dict,
        ) )

        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})").format(
                epoch_ndx,
                mode_str + '_neg',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            ))



        #need to flag everything that looks like they might be part of a nodule
        #semantic segmentation - classifying individual pixels 







