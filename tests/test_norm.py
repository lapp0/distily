import torch
import distily
import pytest


torch.manual_seed(42)
INPUT_TENSORS = [
    torch.randn(100, 20),
    torch.randn(250, 50),
    torch.randn(400, 25),
    torch.randn(1000, 10),
    torch.randn(1000, 100),
    torch.randn(10000, 10),
    torch.randn(10000, 100),
    torch.randn(10000, 1000),
]


DTYPES = [torch.float64, torch.float32]
if torch.cuda.is_available():
    DTYPES += [torch.bfloat16, torch.float16]


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_centering(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    centered_mean = whitened.mean(dim=0)
    torch.testing.assert_close(centered_mean, torch.zeros_like(centered_mean), rtol=0, atol=1e-5)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_covariance(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)
    identity = torch.eye(whitened.size(1), dtype=dtype)

    torch.testing.assert_close(cov_whitened, identity, rtol=0, atol=1e-4)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_eigenvalue_decomposition(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    # Covariance matrix for the centered and whitened input
    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)

    # The covariance matrix of the whitened features should be close to the identity matrix
    identity_matrix = torch.eye(cov_whitened.size(0), device=cov_whitened.device, dtype=cov_whitened.dtype)
    torch.testing.assert_close(cov_whitened, identity_matrix, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_whitening_transformation(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)
    identity = torch.eye(whitened.size(1), dtype=dtype)

    torch.testing.assert_close(cov_whitened, identity, rtol=0, atol=1e-4)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_preservation_of_dimensionality(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    assert x.shape == whitened.shape, "Dimensionality of the output should match the input"


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_whitening_properties(x, dtype):
    x = x.to(dtype)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    # Check that the mean of the whitened data is close to zero
    centered_mean = whitened.mean(dim=0)
    torch.testing.assert_close(centered_mean, torch.zeros_like(centered_mean), rtol=0, atol=1e-5)

    # Check that the covariance matrix of the whitened data is close to the identity matrix
    covariance_matrix = torch.mm(whitened.T, whitened) / (whitened.shape[0] - 1)
    identity_matrix = torch.eye(whitened.size(1), dtype=dtype)
    torch.testing.assert_close(covariance_matrix, identity_matrix, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_numerical_stability(x, dtype):
    x = x.to(dtype) * 1e-10  # Very small values to test stability
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    assert not torch.isnan(whitened).any(), "There should be no NaNs in the output"
    assert not torch.isinf(whitened).any(), "There should be no infinities in the output"

    mean_of_whitened = whitened.mean(dim=0)
    torch.testing.assert_close(mean_of_whitened, torch.zeros_like(mean_of_whitened), rtol=0, atol=1e-5)


@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_whitening_gradient_flow(x, dtype):
    x = x.to(dtype).requires_grad_(True)
    whitening = distily.objectives.norm.Whitening1d().to(dtype)
    whitened = whitening(x)

    # Arbitrary loss to test backpropagation
    loss = whitened.sum()

    # Perform backward pass
    loss.backward()

    # Check that gradients are not None
    assert x.grad is not None, "Gradients should not be None."

    # Check that gradients are finite
    assert torch.isfinite(x.grad).all(), "All gradients should be finite."
