import pytest

from src.interface.layer import Layer
from src.interface.pytorch.nodes.activations import (
    ELU,
    Hardswish,
    LeakyReLU,
    PReLU,
    ReLU,
    Sigmoid,
    Tanh,
)
from src.interface.pytorch.nodes.conv import Conv2d, UpConv2d
from src.interface.pytorch.nodes.drop import Dropout, Dropout2d
from src.interface.pytorch.nodes.linear import Linear
from src.interface.pytorch.nodes.norm import BatchNorm2d, InstanceNorm2d, LayerNorm
from src.interface.pytorch.nodes.pool import AveragePool2d, MaxPool2d
from src.interface.pytorch.nodes.pad import ZeroPadding, ReplicationPadding, ReflectionPadding

P = 0.33


def use_layer_methods(layer: Layer) -> None:
    instance = layer(input_shape=(1, 128, 128))
    print("")
    print(instance)
    print(instance.torch)
    instance.create()
    assert instance.torch is not None
    print("Clone:")
    print(instance.clone())
    print("Mutated:")
    print(instance.mutate(P))


@pytest.mark.spec
class TestPool:
    def test_max2d(self) -> None:
        use_layer_methods(MaxPool2d)

    def test_avg2d(self) -> None:
        use_layer_methods(AveragePool2d)


@pytest.mark.spec
class TestConv:
    def test_conv2d(self) -> None:
        use_layer_methods(Conv2d)

    def test_upconv2d(self) -> None:
        use_layer_methods(UpConv2d)


@pytest.mark.spec
class TestNorm:
    def test_bnorm2d(self) -> None:
        use_layer_methods(BatchNorm2d)

    def test_instancenorm2d(self) -> None:
        use_layer_methods(InstanceNorm2d)

    def test_layernorm(self) -> None:
        use_layer_methods(LayerNorm)

    # def test_groupnorm(self) -> None:
    #     use_layer_methods(GroupNorm)


@pytest.mark.spec
class TestDrop:
    def test_dropout(self) -> None:
        use_layer_methods(Dropout)

    def test_dropout2d(self) -> None:
        use_layer_methods(Dropout2d)


@pytest.mark.spec
class TestLinear:
    def test_linear(self) -> None:
        use_layer_methods(Linear)


@pytest.mark.spec
class TestActivations:
    def test_ReLU(self) -> None:
        use_layer_methods(ReLU)

    def test_PReLU(self) -> None:
        use_layer_methods(PReLU)

    def test_Hardswish(self) -> None:
        use_layer_methods(Hardswish)

    def test_ELU(self) -> None:
        use_layer_methods(ELU)

    def test_LeakyReLU(self) -> None:
        use_layer_methods(LeakyReLU)

    def test_Sigmoid(self) -> None:
        use_layer_methods(Sigmoid)

    def test_Tanh(self) -> None:
        use_layer_methods(Tanh)


@pytest.mark.spec
class TestPad:
    def test_ZeroPad(self) -> None:
        use_layer_methods(ZeroPadding)

    def test_ReplicationPadding(self) -> None:
        use_layer_methods(ReplicationPadding)

    def test_ReflectionPadding(self) -> None:
        use_layer_methods(ReflectionPadding)
