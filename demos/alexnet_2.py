import torch
import torch.nn as nn
import torchyield as ty

def alexnet():
    # `make_layers()` zips together all the keyword arguments 
    # (including any scalar ones), then calls the same factory 
    # function on each set of arguments.  The `channels()` helper 
    # is useful: it breaks a list of channels into separate lists 
    # of input and output channels.  Also note that this factory 
    # function is smart enough to skip the pooling module when the 
    # `pool_size` parameter is 1 (or less).
    yield from ty.make_layers(
            ty.conv2_relu_maxpool_layer,
            **ty.channels([3, 96, 256, 384, 384, 256]),
            kernel_size=[11, 5, 3, 3, 3],
            stride=[4, 1, 1, 1, 1],
            padding=[0, 2, 1, 1, 1],
            pool_size=[3, 3, 1, 1, 3],
            pool_stride=2,
    )

    yield nn.Flatten()

    # `mlp_layer()` is very similar to `make_layers()`.  The only 
    # difference is that instead of using the given factory to 
    # make the last layer, it just makes a plain linear layer. 
    # This is because you typically don't want any nonlinear/ 
    # regularization layers after the last linear layer.
    yield from ty.mlp_layer(
            ty.linear_relu_dropout_layer,
            **ty.channels([36 * 256, 4096, 4096, 1000])
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
