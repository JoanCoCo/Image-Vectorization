import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm

class Flattener():
    def __init__(self, palette : np.array, src_image : str, display_progress_bar : bool = False) -> None:
        img = Image.open(src_image).convert("RGBA")
        self.reference = torch.tensor(np.array(img).astype(float) / 255.0)
        self.palette = torch.tensor(palette)
        self.flat_image = self.reference[:, :, 0:3].clone().detach() #torch.rand_like(self.reference[:, :, 0:3], requires_grad=True)
        self.flat_image.requires_grad = True
        self.progress_bar = display_progress_bar

    def model_filter(self, image : Image, iterations : int = 5) -> Image:
        result = image.filter(ImageFilter.ModeFilter)
        for _ in range(iterations):
            result = result.filter(ImageFilter.ModeFilter)
        return result

    def get_distance_with_the_palette(self) -> torch.Tensor:
        nfm = self.flat_image
        color_layers = torch.tile(self.palette[:, None, None, ...], (1, self.reference.shape[0], self.reference.shape[1], 1))
        my_dists = torch.norm(torch.tile(nfm[None, ...], (self.palette.shape[0], 1, 1, 1)) - color_layers, p=2, dim=-1)
        my_dists = torch.min(my_dists, dim=0)[0]
        return my_dists

    def palette_loss(self) -> torch.Tensor:
        return torch.mean(self.get_distance_with_the_palette())
    
    def image_loss(self) -> torch.Tensor:
        nfm = self.flat_image
        return torch.mean(torch.norm(self.reference[:, :, 0:3] - nfm, p=2, dim=-1))

    def optimize(self, iterations : int, lr : float = 0.01, verbose : int = 0):
        optimizer = torch.optim.Adam([self.flat_image], lr=lr)
        for it in tqdm(range(iterations), ncols=60, disable=not self.progress_bar, bar_format="|{bar}|{desc}: {percentage:3.0f}%"):
            optimizer.zero_grad()
            pal = self.palette_loss()
            img = self.image_loss()
            loss = pal + img
            if verbose > 0 and it % verbose == 0:
                print("Iteration {:d}:".format(it))
                print("\tPalette loss: {:.5f}".format(pal.item()))
                print("\tImage loss: {:.5f}".format(img.item()))
                print("\tTotal loss: {:.5f}".format(loss.item()))
            loss.backward()
            optimizer.step()
            self.flat_image.data.clamp_(0.0, 1.0)

    def get_flat_image(self) -> np.array:
        return torch.concat([self.flat_image, self.reference[:, :, 3][..., None]], dim=-1).detach().numpy()
        #return self.deep_flat_image(self.input_batches.float()).reshape(self.reference.shape[0], self.reference.shape[1], 3).detach().numpy()