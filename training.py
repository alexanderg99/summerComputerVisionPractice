import argparse
import sys
import datetime
from logging import log
from torch import nn
from torch.utils.data import DataLoader
from dsets import LunaDataset



#this helps us use the command line
import torch.cuda


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

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self)
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



