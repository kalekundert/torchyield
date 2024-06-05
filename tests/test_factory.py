import torch.nn as nn
import torchyield as ty
import pytest

def test_linear():
    net, = ty.linear_layer(
            in_channels=1,
            out_channels=2,
    )

    assert isinstance(net, nn.Linear)
    assert net.in_features == 1
    assert net.out_features == 2

def test_linear_err_no_in_channel():
    with pytest.raises(
            TypeError,
            match=r"linear_layer\(\) missing required argument. 'in_channels'",
    ):
        _, = ty.linear_layer(
                out_channels=2,
        )

def test_linear_err_no_out_channel():
    with pytest.raises(
            TypeError,
            match=r"linear_layer\(\) missing required argument. 'out_channels'",
    ):
        _, = ty.linear_layer(
                in_channels=1,
        )

def test_conv1():
    net, = ty.conv1_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
    )

    assert isinstance(net, nn.Conv1d)
    assert net.in_channels == 1
    assert net.out_channels == 2
    assert net.kernel_size == (3,)

def test_conv1_all_args():
    net, = ty.conv1_layer(
            in_channels=14,
            out_channels=28,
            kernel_size=3,
            stride=4,
            padding=5,
            dilation=6,
            groups=7,
            bias=False,
            padding_mode='reflect',
    )

    assert isinstance(net, nn.Conv1d)
    assert net.in_channels == 14
    assert net.out_channels == 28
    assert net.kernel_size == (3,)
    assert net.stride == (4,)
    assert net.padding == (5,)
    assert net.dilation == (6,)
    assert net.groups == 7
    assert net.bias is None
    assert net.padding_mode == 'reflect'

def test_conv2():
    net, = ty.conv2_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
    )

    assert isinstance(net, nn.Conv2d)
    assert net.in_channels == 1
    assert net.out_channels == 2
    assert net.kernel_size == (3, 3)

def test_conv3():
    net, = ty.conv3_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
    )

    assert isinstance(net, nn.Conv3d)
    assert net.in_channels == 1
    assert net.out_channels == 2
    assert net.kernel_size == (3, 3, 3)

def test_maxpool_err():
    with pytest.raises(
            ValueError,
            match=r"'maxpool' must come after 'linear' or 'conv'",
    ):
        _, = ty.maxpool_layer()

def test_avgpool_err():
    with pytest.raises(
            ValueError,
            match=r"'avgpool' must come after 'linear' or 'conv'",
    ):
        _, = ty.avgpool_layer()

def test_relu():
    net, = ty.relu_layer()

    assert isinstance(net, nn.ReLU)
    assert net.inplace == True

def test_relu_inplace():
    net, = ty.relu_layer(inplace=False)

    assert isinstance(net, nn.ReLU)
    assert net.inplace == False

def test_bn_err():
    with pytest.raises(
            ValueError,
            match=r"'bn' must come after 'linear' or 'conv'",
    ):
        _, = ty.bn_layer()

def test_dropout():
    net, = ty.dropout_layer(
            dropout_p=0.1,
    )
    assert isinstance(net, nn.Dropout)
    assert net.p == 0.1


def test_linear_relu():
    linear, relu = ty.linear_relu_layer(
            in_channels=1,
            out_channels=2,
    )

    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 1
    assert linear.out_features == 2

    assert isinstance(relu, nn.ReLU)

def test_linear_relu_dropout():
    linear, relu, dropout = ty.linear_relu_dropout_layer(
            in_channels=1,
            out_channels=2,
            dropout_p=0.3,
    )

    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 1
    assert linear.out_features == 2

    assert isinstance(relu, nn.ReLU)

    assert isinstance(dropout, nn.Dropout)
    assert dropout.p == 0.3

def test_conv2_bn():
    conv, bn = ty.conv2_bn_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            bn_eps=1e-4,
            bn_momentum=0.2,
            bn_affine=False,
            bn_track_running_stats=False,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)
    assert conv.bias is None

    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.num_features == 2
    assert bn.eps == 1e-4
    assert bn.momentum == 0.2
    assert bn.affine == False
    assert bn.track_running_stats == False

def test_conv2_relu_bn():
    conv, relu, bn = ty.conv2_relu_bn_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            bn_eps=1e-4,
            bn_momentum=0.2,
            bn_affine=False,
            bn_track_running_stats=False,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)
    assert conv.bias is not None

    assert isinstance(relu, nn.ReLU)

    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.num_features == 2
    assert bn.eps == 1e-4
    assert bn.momentum == 0.2
    assert bn.affine == False
    assert bn.track_running_stats == False

def test_conv2_maxpool():
    conv, pool = ty.conv2_maxpool_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,

            pool_size=4,
            pool_stride=5,
            pool_padding=6,
            pool_dilation=7,
            pool_ceil_mode=True,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)

    assert isinstance(pool, nn.MaxPool2d)
    assert pool.kernel_size == 4
    assert pool.stride == 5
    assert pool.padding == 6
    assert pool.dilation == 7
    assert pool.ceil_mode == True

def test_conv2_maxpool_skip():
    conv, = ty.conv2_maxpool_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,

            pool_size=1,
            pool_stride=5,
            pool_padding=6,
            pool_dilation=7,
            pool_ceil_mode=True,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)

def test_conv2_avgpool():
    conv, pool = ty.conv2_avgpool_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,

            pool_size=4,
            pool_stride=5,
            pool_padding=6,
            pool_ceil_mode=True,
            pool_count_include_pad=False,
            pool_divisor_override=7,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)

    assert isinstance(pool, nn.AvgPool2d)
    assert pool.kernel_size == 4
    assert pool.stride == 5
    assert pool.padding == 6
    assert pool.ceil_mode == True
    assert pool.count_include_pad == False
    assert pool.divisor_override == 7

def test_conv2_avgpool_skip():
    conv, = ty.conv2_avgpool_layer(
            in_channels=1,
            out_channels=2,
            kernel_size=3,

            pool_size=1,
            pool_stride=5,
            pool_padding=6,
            pool_ceil_mode=True,
            pool_count_include_pad=False,
            pool_divisor_override=7,
    )

    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 1
    assert conv.out_channels == 2
    assert conv.kernel_size == (3, 3)


def test_err_unknown_module():
    with pytest.raises(
            AttributeError,
            match=r"linear_relu_max_pool_layer\(\) includes unknown module 'max'\n.*did you mean. 'maxpool'",
    ):
        _, _, _ = ty.linear_relu_max_pool_layer(
                in_channels=1,
                out_channels=2,
        )

def test_err_unused_arg():
    with pytest.raises(
            TypeError,
            match=r"linear_layer\(\) got unexpected keyword argument\(s\). 'unused_arg'",
    ):
        _, = ty.linear_layer(
                in_channels=1,
                out_channels=2,
                unused_arg=3,
        )


