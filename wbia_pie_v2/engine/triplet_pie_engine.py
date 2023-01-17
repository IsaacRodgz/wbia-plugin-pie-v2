# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from torch_ort import ORTModule
import torch
import metrics
from losses import TripletLoss, CrossEntropyLoss

from engine import PIEEngine


class TripletPIEEngine(PIEEngine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of datamanager.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer.
                Default is True.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
    ):
        super(TripletPIEEngine, self).__init__(datamanager, use_gpu)

        #self.model = ORTModule(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth,
        )
        print('***Initialized Triplet PIE Engine***')

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.model.module.__class__.__name__ == 'ViTReid':
          imgs = [self.model.module.topil(img) for img in imgs]
          imgs = self.model.module.feature_extractor(imgs, return_tensors='pt')
          imgs = imgs['pixel_values']

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        with torch.cuda.amp.autocast():
          outputs, features = self.model(imgs)

          loss = 0
          loss_summary = {}

          if self.weight_t > 0:
              loss_t = self.compute_loss(self.criterion_t, features, pids)
              loss += self.weight_t * loss_t
              loss_summary['loss_t'] = loss_t.item()

          if self.weight_x > 0:
              loss_x = self.compute_loss(self.criterion_x, outputs, pids)
              loss += self.weight_x * loss_x
              loss_summary['loss_x'] = loss_x.item()
              loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad(set_to_none=True)
        #loss.backward()
        self.scaler.scale(loss).backward()
        #self.optimizer.step()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss_summary
