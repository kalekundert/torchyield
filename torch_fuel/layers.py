import torch.nn as nn

from more_itertools import zip_broadcast, pairwise, unzip
from itertools import cycle

from collections.abc import Iterable, Callable
from typing import TypeAlias

LayerFactory: TypeAlias = Callable[..., Iterable[nn.Module]]

BRIGHT_WHITE = '\033[97m'
RESET_COLOR = '\033[0m'
DEFAULT_VERBOSE_TEMPLATE = f'{BRIGHT_WHITE}{{}}\n{{}}{RESET_COLOR}\n{79*"─"}'

class Layers(nn.Sequential):

    def __init__(self, *layers, verbose=False):
        layers = concat_layers(*layers)

        if verbose:
            layers = globals()['verbose'](layers)

        super().__init__(*layers)

class VerboseModuleWrapper(nn.Module):

    def __init__(self, module, template=DEFAULT_VERBOSE_TEMPLATE, **kwargs):
        super().__init__()
        self.module = module
        self.template = template
        self.print_kwargs = kwargs

    def forward(self, x):
        print(self.template.format(self.module, x.shape), **self.print_kwargs)
        return self.module(x)


def make_layers(layer_factory: LayerFactory, **params):
    # Normally we want to be strict, but `itertools.cycle()` is useful enough 
    # to merit an exception.
    strict = not any(isinstance(x, cycle) for x in params.values())

    for values in zip_broadcast(*params.values(), strict=strict):
        kwargs = dict(zip(params.keys(), values))
        yield from layer_factory(**kwargs)

def concat_layers(*layers: nn.Module | Iterable[nn.Module]):
    for layer in layers:
        if isinstance(layer, nn.Module):
            yield layer
        else:
            yield from layer

def verbose(layers):
    yield from map(VerboseModuleWrapper, layers)

def channels(
        channels: Iterable[int],
        keys: tuple[str, str] = ('in_channels', 'out_channels'),
) -> dict[str, list[int]]:
    values = map(list, unzip(pairwise(channels)))
    return dict(zip(keys, values))


def linear_relu_layer(
        *,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
):
    yield nn.Linear(in_channels, out_channels, bias=True)
    yield nn.ReLU()

def linear_relu_dropout_layer(
        *,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        drop_rate: float,
):
    assert 0 <= drop_rate <= 1
    yield nn.Linear(in_channels, out_channels, bias=True)
    yield nn.ReLU()
    yield nn.Dropout(drop_rate)

