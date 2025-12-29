from model.utils.logger import logger
from collections import OrderedDict, namedtuple
from torch import nn

from model.encoder.cosmos import Encoder
from model.decoder.cosmos import Decoder

NetworkEval = namedtuple("NetworkEval", ["reconstructions"])

class DiscreteImageTokenizer(nn.Module):
    def __init__(self, z_channels: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "DiscreteImageTokenizer")
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(z_channels=z_channels, embedding_dim=embedding_dim, **kwargs)
        self.decoder = Decoder(z_channels=z_channels, embedding_dim=embedding_dim, **kwargs)


        num_parameters = sum(p.numel() for p in self.parameters())
        logger.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logger.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

    def encode(self, x):
        z = self.encoder(x)     # (B, embedding_dim, H', W')
        return z

    def decode(self, z): 
        x_hat = self.decoder(z) # (B, embedding_dim, H', W')
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    



if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = dict(
        attn_resolutions=[6, 12],
        channels=128,
        channels_mult=[2, 4, 4],
        dropout=0.0,
        in_channels=1,
        spatial_compression=8,
        num_res_blocks=2,
        out_channels=1,
        resolution=96,
        patch_size=2,
        patch_method="haar",
        z_channels=256,
        z_factor=2,
        embedding_dim=64,
        num_embeddings=8192,
        num_quantizers=1,
        name="DI",
    )

    model = DiscreteImageTokenizer(**params)
    model.to(device)
    model.eval()

    # fake input image: (B, C, H, W)
    x = torch.randn(1, 1, 256, 256, device=device)

    with torch.no_grad():
        x_hat = model(x)

    print("input shape :", x.shape)
    print("output shape:", x_hat.shape)