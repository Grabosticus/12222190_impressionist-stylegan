import torch
from generator import Generator
from discriminator import Discriminator
import globals

def test_generator_output_shape_4x4_res():
    G = Generator().to("cpu")
    z = torch.randn(4, globals.Z_DIM)
    img = G(z)

    assert img.shape == (4, 3, 4, 4)
    assert img.dtype == torch.float32

def test_generator_output_shape_after_fade_in_8x8_res():
    G = Generator().to("cpu")
    G.fade_in(8)
    z = torch.randn(4, globals.Z_DIM)
    img = G(z)

    assert img.shape == (4, 3, 8, 8)
    assert img.dtype == torch.float32

def test_generator_value_range():
    G = Generator().to("cpu")
    z = torch.randn(2, globals.Z_DIM)
    img = G(z)

    assert torch.isfinite(img).all()
    assert img.min() >= -1
    assert img.max() <= 1

def test_discriminator_output_shape():
    D = Discriminator().to("cpu")
    x = torch.randn(4, 3, 4, 4) # 4 4x4 RGB images
    out = D(x)

    assert out.ndim == 2
    assert out.shape[0] == 4
    assert torch.isfinite(out).all()

print("GENERATOR TESTS")
test_generator_output_shape_4x4_res()
test_generator_output_shape_after_fade_in_8x8_res()
test_generator_value_range()
print("Passed.")
print("DISCRIMINATOR TESTS")
test_discriminator_output_shape()
print("Passed.")