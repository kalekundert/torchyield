import torch.nn as nn
import torchyield as ty
import pytest

from itertools import cycle
from more_itertools import UnequalIterablesError

def test_layers_cnn():

    def conv_relu_maxpool(in_channels, out_channels, kernel_size, pool_size):
        yield nn.Conv2d(in_channels, out_channels, kernel_size)
        yield nn.ReLU()
        yield nn.MaxPool2d(pool_size)

    def linear_relu(in_channels, out_channels):
        yield nn.Linear(in_channels, out_channels)
        yield nn.ReLU()

    cnn = ty.Layers(
            ty.make_layers(
                conv_relu_maxpool,
                **ty.channels([3, 32, 64]),
                kernel_size=5,
                pool_size=[1, 2],
            ),
            nn.Flatten(),
            ty.make_layers(
                linear_relu,
                **ty.channels([64*4, 1]),
            ),
    )

    assert len(cnn) == 9

    assert isinstance(cnn[0], nn.Conv2d)
    assert cnn[0].in_channels == 3
    assert cnn[0].out_channels == 32
    assert cnn[0].kernel_size == (5, 5)

    assert isinstance(cnn[1], nn.ReLU)

    assert isinstance(cnn[2], nn.MaxPool2d)
    assert cnn[2].kernel_size == 1

    assert isinstance(cnn[3], nn.Conv2d)
    assert cnn[3].in_channels == 32
    assert cnn[3].out_channels == 64
    assert cnn[3].kernel_size == (5, 5)

    assert isinstance(cnn[4], nn.ReLU)

    assert isinstance(cnn[5], nn.MaxPool2d)
    assert cnn[5].kernel_size == 2

    assert isinstance(cnn[6], nn.Flatten)

    assert isinstance(cnn[7], nn.Linear)
    assert cnn[7].in_features == 64 * 4
    assert cnn[7].out_features == 1

    assert isinstance(cnn[8], nn.ReLU)


def test_make_layers():

    def layer_factory(**kwargs):
        yield kwargs

    layers = ty.make_layers(
            layer_factory,
            a=1,
            b=[2,3],
    )
    assert list(layers) == [
            {'a': 1, 'b': 2},
            {'a': 1, 'b': 3},
    ]

def test_make_layers_strict():

    def layer_factory(**kwargs):
        yield kwargs

    layers = ty.make_layers(
            layer_factory,
            a=[1,2],
            b=[3,4,5],
    )
    with pytest.raises(UnequalIterablesError):
        list(layers)

def test_make_layers_cycle():

    def layer_factory(**kwargs):
        yield kwargs

    layers = ty.make_layers(
            layer_factory,
            a=[1,2,3,4],
            b=cycle([5,6]),
    )
    assert list(layers) == [
            {'a': 1, 'b': 5},
            {'a': 2, 'b': 6},
            {'a': 3, 'b': 5},
            {'a': 4, 'b': 6},
    ]


def test_channels():
    assert ty.channels([1,2,3,4]) == dict(
            in_channels=[1,2,3],
            out_channels=[2,3,4],
    )

def test_channels_keys():
    assert ty.channels([1,2,3,4], keys=('a', 'b')) == dict(
            a=[1,2,3],
            b=[2,3,4],
    )

