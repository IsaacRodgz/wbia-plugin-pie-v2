# -*- coding: utf-8 -*-

import logging
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as T

logger = logging.getLogger(__name__)

class ViTReid(nn.Module):
    """Re-id model with ViT as a Transformer feature extractor.
    Input:
        core_name (string): name of core model, class from HF transformers library
    """

    def __init__(self, core_name, num_classes, fc_dims, dropout_p, loss):
        super().__init__()
        self.loss = loss

        self.topil = T.ToPILImage()

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(core_name)
        self.core_model = ViTModel.from_pretrained(core_name)

        self.fc = self._construct_fc_layer(
            fc_dims, 768, dropout_p
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def forward(self, x):
        #if self.training:
        #  import pdb; pdb.set_trace()
        #x = [self.topil(img) for img in x]
        #x = self.feature_extractor(x, return_tensors='pt')
        #x = x['pixel_values']
        v = self.core_model(x).pooler_output

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if 'softmax' in self.loss:
            return y
        elif 'triplet' in self.loss:
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def vit(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ViTReid(
        core_name='google/vit-base-patch16-224-in21k',
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        dropout_p=None,
    )

    return model