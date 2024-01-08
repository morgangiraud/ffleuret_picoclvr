import torch

import mygpt


# print(f"a: {a}")
# print(f"slice_out: {slice_out}")
# print(f"complete_out: {complete_out}")


def test_sanity():
    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, 5))

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=4,
        dim_keys=2,
        dim_hidden=2,
        nb_heads=2,
        nb_blocks=2,
        dropout=0.1,
        causal=True,
    )

    model.eval()
    y1 = model(mygpt.BracketedSequence(x)).x
    y2 = torch.randn_like(y1)
    for s in range(x.size(1)):
        z = model(mygpt.BracketedSequence(x, s, 1))
        y2[:, s] = z.slice()

    error = ((y1 - y2).norm() / (y1.norm() + y2.norm())).item()
    print(f"error: {error}")
    assert error < 1e-7
