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
    torch.randn(800, 800),
]


DTYPES = [torch.float64, torch.float32]
if torch.cuda.is_available():
    DTYPES += [torch.bfloat16, torch.float16]


WHITENING_MODULES = [
    distily.objectives.norm.Whitening1dZCA,
    distily.objectives.norm.Whitening1dCholesky,
    distily.objectives.norm.Whitening1dSVD
]


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_identity_covariance_matrix(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)
    identity = torch.eye(whitened.size(1), dtype=dtype)

    torch.testing.assert_close(cov_whitened, identity, rtol=0, atol=1e-4)


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_mean(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    centered_mean = whitened.mean(dim=0)
    torch.testing.assert_close(centered_mean, torch.zeros_like(centered_mean), rtol=0, atol=1e-5)


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_deterministic_transformation(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)

    # Ensure deterministic output
    whitened1 = whitening(x)
    whitened2 = whitening(x)
    torch.testing.assert_close(whitened1, whitened2, rtol=0, atol=0)


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_numerical_stability(whitener, x, dtype):
    x = x.to(dtype) * 1e-10  # Very small values to test stability
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    assert not torch.isnan(whitened).any(), "There should be no NaNs in the output"
    assert not torch.isinf(whitened).any(), "There should be no infinities in the output"

    mean_of_whitened = whitened.mean(dim=0)
    torch.testing.assert_close(mean_of_whitened, torch.zeros_like(mean_of_whitened), rtol=0, atol=1e-5)


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_whitening_gradient_flow(whitener, x, dtype):
    x = x.to(dtype).requires_grad_(True)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    # Arbitrary loss to test backpropagation
    loss = whitened.sum()

    # Perform backward pass
    loss.backward()

    # Check that gradients are not None
    assert x.grad is not None, "Gradients should not be None."

    # Check that gradients are finite
    assert torch.isfinite(x.grad).all(), "All gradients should be finite."


@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_eigenvalue_spectrum(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    # Compute the covariance matrix of the whitened data
    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)

    # Compute the eigenvalues of the covariance matrix
    eigenvalues, _ = torch.linalg.eigh(cov_whitened)

    # Check if all eigenvalues are close to 1
    torch.testing.assert_close(eigenvalues, torch.ones_like(eigenvalues), rtol=1e-4, atol=1e-4)


@pytest.mark.skip("Not confident linear transformation guarantee is necessary")
@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_transformation(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)

    # Create a scaled version of the input
    scaling_factor = 2.5
    scaled_x = scaling_factor * x

    # Apply whitening to both original and scaled inputs
    whitened = whitening(x)
    whitened_scaled = whitening(scaled_x)

    # Verify that the whitening transformation is linear
    torch.testing.assert_close(whitened_scaled, scaling_factor * whitened, rtol=1e-4, atol=1e-4)


@pytest.mark.skip("Not confident invertibility guarantee is necessary")
@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_invertibility(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    # Infer the whitening matrix from the output and the pseudoinverse of the input
    x_pseudo_inv = torch.pinverse(x.T)
    inferred_whitening_matrix = torch.mm(whitened.T, x_pseudo_inv)

    # Check whether the whitening and its inverse recover the original data
    dewhitened = torch.mm(whitened, torch.pinverse(inferred_whitening_matrix.T))
    torch.testing.assert_close(x, dewhitened, rtol=1e-4, atol=1e-4)


@pytest.mark.skip("Not confident sequential decorrelation guarantee is necessary")
@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sequential_decorrelation(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    # Check sequential decorrelation: each feature should be uncorrelated with the preceding ones
    cov_matrix = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)
    assert torch.all(torch.tril(cov_matrix, diagonal=-1) == 0), "Features should be sequentially decorrelated"


@pytest.mark.skip("Not confident avoidance of rotations guarantee is necessary")
@pytest.mark.parametrize("whitener", WHITENING_MODULES)
@pytest.mark.parametrize("x", INPUT_TENSORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_no_rotation_of_data(whitener, x, dtype):
    x = x.to(dtype)
    whitening = whitener().to(dtype)
    whitened = whitening(x)

    # Compute eigenvectors of the covariance matrix of original data
    cov_original = torch.mm(x.T, x) / (x.size(0) - 1)
    _, eigvecs_original = torch.linalg.eigh(cov_original)

    # Compute eigenvectors of the covariance matrix of whitened data
    cov_whitened = torch.mm(whitened.T, whitened) / (whitened.size(0) - 1)
    _, eigvecs_whitened = torch.linalg.eigh(cov_whitened)

    # Check if eigenvectors align, indicating no rotation
    assert torch.allclose(eigvecs_original.abs(), eigvecs_whitened.abs(), atol=1e-2)
