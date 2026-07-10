import torch

from models.pet import PETBlock, PETEncoder


def test_pet_block_output_shape_single_event():
    block = PETBlock(d=16, k=4)
    x = torch.randn(20, 16)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    assert out.shape == (20, 16)


def test_pet_block_output_shape_multi_event(tiny_pyg_batch):
    d = 16
    n = tiny_pyg_batch.x.shape[0]
    block = PETBlock(d=d, k=4)
    x = torch.randn(n, d)
    out = block(x, tiny_pyg_batch.batch)
    assert out.shape == (n, d)


def test_pet_block_gradients_flow_to_all_params():
    block = PETBlock(d=16, k=4)
    x = torch.randn(20, 16, requires_grad=True)
    batch = torch.zeros(20, dtype=torch.long)
    out = block(x, batch)
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in block.parameters())


def test_pet_block_handles_k_larger_than_event_size():
    # event has fewer than k=8 nodes; knn_graph must not error
    block = PETBlock(d=16, k=8)
    x = torch.randn(5, 16)
    batch = torch.zeros(5, dtype=torch.long)
    out = block(x, batch)
    assert out.shape == (5, 16)


def test_pet_encoder_forward_shape(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    out = encoder(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    assert out.shape == (tiny_pyg_batch.x.shape[0], 16)


def test_pet_encoder_encode_event_shape(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    n_events = int(tiny_pyg_batch.batch.max().item()) + 1
    g = encoder.encode_event(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    assert g.shape == (n_events, 16)


def test_pet_encoder_gradients_flow(tiny_pyg_batch):
    encoder = PETEncoder(d=16, L=2, k=4)
    g = encoder.encode_event(tiny_pyg_batch.x, tiny_pyg_batch.batch)
    g.sum().backward()
    assert all(p.grad is not None for p in encoder.parameters())
