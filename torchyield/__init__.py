"""
Utilities for working with PyTorch.
"""

__version__ = '0.0.0'

from .layers import *
from .verbose import *

def __getattr__(name):
    """
    Dynamically create layer factories.

    For example, `ty.linear_bn_relu_layer()` is a factory that yields a linear 
    layer, followed by a batch normalization layer, followed by a ReLU.  The 
    rules for creating a factory are as follows:

    - The function name must end in "_layer".

    - The rest of the function name is a underscore-separated list of modules 
      to include in the layer.  The following modules can be specified:

      ========  =============
      Name      Module
      ========  =============
      linear    nn.Linear
      conv1     nn.Conv1d
      conv2     nn.Conv2d
      conv3     nn.Conv3d
      relu      nn.ReLU
      bn        nn.BatchNorm*
      dropout   nn.DropOut
      ========  =============

    - The resulting factory will accept arguments for each module it creates.  
      For linear and convolutional modules, the argument names are the same as 
      for the corresponding modules.  For all other modules, the argument names 
      are prefixed by the name of the module (e.g. a dropout rate can be 
      specified as: `linear_dropout(dropout_p=0.1)`).

    Special considerations:

    - The input and output dimensions for linear layers are specified via 
      *in_channels* and *out_channels* arguments, not *in_features* and 
      *out_features* as expected by `nn.Linear`.  This is just for consistency 
      with the convolutional layers, and to make the `channels()` helper 
      function easier to use.

    - If a convolutional or linear layer is followed immediately by a batch 
      norm layer, the bias will be disabled by default.  Since the batch norm 
      will re-center the output on 0 anyways, there's no reason to calculate a 
      bias in such cases.

    - Batch norm layers must be preceded by a linear or convolutional layer, 
      and the dimensionality of the batch norm is determined by that.
    """
    if not name.endswith('_layer'):
        raise AttributeError(name)

    module_names = name.split('_')[:-1]

    conv_factories = {
            'conv1': nn.Conv1d,
            'conv2': nn.Conv2d,
            'conv3': nn.Conv3d,
    }
    conv_dimensions = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 3,
    }
    bn_factories = {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
    }

    def factory(**kwargs):
        curr_channels = None
        curr_dimensions = None
        kwargs_used = set()

        for i, module_name in enumerate(module_names):

            def pop_channels(alias='channels'):
                nonlocal curr_channels
                kwargs_used.update(['in_channels', 'out_channels'])

                try:
                    curr_channels = kwargs.pop('out_channels')
                    return {
                            f'in_{alias}': kwargs.pop('in_channels'),
                            f'out_{alias}': curr_channels,
                    }
                except KeyError as err:
                    raise TypeError(f"{name}() missing required argument: {err}")

            def get_bias():
                kwargs_used.add('bias')

                try:
                    return kwargs['bias']
                except KeyError:
                    pass

                try:
                    return module_names[i + 1] != 'bn'
                except IndexError:
                    return True

            def get_kwargs(*keys, prefix=''):
                out = {}
                for k in keys:
                    if (pk := prefix + k) in kwargs:
                        out[k] = kwargs[pk]
                        kwargs_used.add(pk)
                return out

            if module_name == 'linear':
                curr_dimensions = 1
                yield nn.Linear(
                        **pop_channels('features'),
                        bias=get_bias(),
                )

            elif module_name in conv_factories:
                curr_dimensions = conv_dimensions[module_name]
                yield conv_factories[module_name](
                        **pop_channels(),
                        **get_kwargs(
                            'kernel_size',
                            'stride',
                            'padding',
                            'dilation',
                            'groups',
                            'padding_mode',
                        ),
                        bias=get_bias(),
                )

            elif module_name == 'relu':
                yield nn.ReLU(inplace=True)

            elif module_name == 'bn':
                if not curr_dimensions:
                    raise ValueError("'bn' must come after 'linear' or 'conv'")

                yield bn_factories[curr_dimensions](
                        curr_channels,
                        **get_kwargs(
                            'eps',
                            'momentum',
                            'affine',
                            'track_running_stats',
                            prefix='bn_',
                        ),
                )

            elif module_name == 'dropout':
                yield nn.Dropout(
                        **get_kwargs(
                            'p',
                            prefix='dropout_',
                        ),
                )

            else:
                raise ValueError(f"unknown layer: {module_name}")

        kwargs_unused = set(kwargs) - kwargs_used
        if kwargs_unused:
            raise TypeError(f"{name}() got unexpected keyword argument(s): {','.join(map(repr, kwargs_unused))}")

    factory.__name__ = name
    factory.__qualname__ = f'torchyield.{name}'
    factory.__module__ = 'torchyield'

    return factory









