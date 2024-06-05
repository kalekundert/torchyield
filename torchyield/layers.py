import torch.nn as nn

from more_itertools import zip_broadcast, pairwise, unzip
from itertools import cycle

from collections.abc import Iterable, Callable
from typing import TypeAlias

LayerFactory: TypeAlias = Callable[..., Iterable[nn.Module] | nn.Module]

class Layers(nn.Sequential):

    def __init__(self, *layers, verbose=False):
        layers = concat_layers(*layers)

        if verbose:
            from .verbose import verbose as _verbose
            layers = _verbose(layers)

        super().__init__(*layers)

def make_layers(layer_factory: LayerFactory, **params):
    # Normally we want to be strict, but `itertools.cycle()` is useful enough 
    # to merit an exception.
    strict = not any(isinstance(x, cycle) for x in params.values())

    for values in zip_broadcast(*params.values(), strict=strict):
        kwargs = dict(zip(params.keys(), values))
        layer = layer_factory(**kwargs)

        if isinstance(layer, nn.Module):
            yield layer
        else:
            yield from layer

def concat_layers(*layers: nn.Module | Iterable[nn.Module]):
    for layer in layers:
        if isinstance(layer, nn.Module):
            yield layer
        else:
            yield from layer

def channels(
        channels: Iterable[int],
        keys: tuple[str, str] = ('in_channels', 'out_channels'),
) -> dict[str, list[int]]:
    values = map(list, unzip(pairwise(channels)))
    return dict(zip(keys, values))


def mlp_layer(layer_factory, in_channels, out_channels, **kwargs):
    yield from make_layers(
            layer_factory,
            in_channels=in_channels[:-1],
            out_channels=out_channels[:-1],
            **kwargs,
    )
    yield nn.Linear(in_channels[-1], out_channels[-1], bias=True)

