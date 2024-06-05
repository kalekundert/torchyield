import torch
import torch.nn as nn
import torchyield as ty

def conv_relu(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
):
    yield nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
    )
    yield nn.ReLU()

def conv_relu_pool(
        pool_size=3,
        pool_stride=2,
        **hparams,
):
    yield from conv_relu(**hparams)
    yield nn.MaxPool2d(
            kernel_size=pool_size,
            stride=pool_stride,
    )

def linear_relu_dropout(
        in_features,
        out_features,
        drop_rate=0.5,
):
    yield nn.Linear(
            in_features=in_features,
            out_features=out_features,
    )
    yield nn.ReLU()
    yield nn.Dropout(p=drop_rate)

def alexnet():
    yield from conv_relu_pool(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
    )
    yield from conv_relu_pool(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            padding=2,
    )
    yield from conv_relu(
            in_channels=256,
            out_channels=384,
            padding=1,
    )
    yield from conv_relu(
            in_channels=384,
            out_channels=384,
            padding=1,
    )
    yield from conv_relu_pool(
            in_channels=384,
            out_channels=256,
            padding=1,
    )

    yield nn.Flatten()

    yield from linear_relu_dropout(
            in_features=36 * 256,
            out_features=4096,
    )
    yield from linear_relu_dropout(
            in_features=4096,
            out_features=4096,
    )
    yield nn.Linear(
            in_features=4096,
            out_features=1000,
    )

if __name__ == '__main__':
    # Convert the generator into an instance of `torch.nn.Sequential`:
    f = ty.Layers(alexnet())

    # Demonstrate that the model works, i.e. it can make a prediction 
    # given random input:
    x = torch.randn(1, 3, 227, 227)
    y = f(x)
    print(torch.argmax(y))  # tensor(388)

    import torchlens
    torchlens.show_model_graph(f, x)
