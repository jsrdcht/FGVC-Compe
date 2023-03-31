import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmcls.structures import ClsDataSample
from mmengine.model import Sequential
from mmengine.model.weight_init import trunc_normal_

from mmcls.registry import MODELS
from .cls_head import ClsHead

class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output

@MODELS.register_module()
class GreedyHashHead(ClsHead):
    """Vision Transformer hash head.

    Args:
        bit (int): Number of bits for hashing.
        num_classes (int): Number of categories excluding the background
            category.
        alpha (float): The weight for the greedyhash loss.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """
    def __init__(self,
                 bit,
                 num_classes,
                 alpha,
                 in_channels,
                 hidden_dim=None,
                 act_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(GreedyHashHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.bit = bit
        self.in_channels = in_channels
        self.criterion = nn.CrossEntropyLoss()

        self._init_layers()

    def _init_layers(self):
        if self.hidden_dim is None:
            layers = [
                ('hash_layer', nn.Linear(self.in_channels, self.bit)),
                ('hash2class', nn.Linear(self.bit, self.num_classes))
                ]
            # layers = [
            #     ('fc_layer', nn.Linear(self.in_channels, self.num_classes)),
            #     ]
            
        else:
            raise NotImplementedError
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        """"Init weights of hidden layer if exists."""
        super(GreedyHashHead, self).init_weights()
        
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

        self.apply(self._init_linear_weights)

    def _init_linear_weights(self, module):
        if isinstance(module, nn.Linear):

            trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pre_logits(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``VisionTransformerClsHead``, we
        obtain the feature of the last stage and forward in hidden layer if
        exists.
        """

        _, cls_token = feats[-1]
        if self.hidden_dim is None:
            return cls_token
        else:
            x = self.layers.pre_logits(cls_token)
            return self.layers.act(x)
        
    def forward(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """Forward function.
        """
        x = self.pre_logits(feats)
        # x = self.layers.fc_layer(x)

        hash_feature = self.layers.hash_layer(x)
        hash_feature = hash_feature.tanh()
        hash_code = Hash.apply(hash_feature)

        gt_pred = self.layers.hash2class(hash_code)

        return gt_pred
    
    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        
        losses = {}

        cls_score = self(feats)

        x = self.pre_logits(feats)
        hash_feature = self.layers.hash_layer(x)
        hash_feature = hash_feature.tanh()

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        # losses is a dict
        losses['cls_loss'] = losses['loss']

        greedy_loss = (hash_feature.abs() - 1).pow(3).abs().mean()
        losses['greedy_loss'] = greedy_loss

        losses['loss'] = losses['cls_loss'] + self.alpha * losses['greedy_loss']
        return losses