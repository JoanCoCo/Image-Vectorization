import numpy as np
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm

class LayerGenerator():
    def __init__(self, image : np.array, palette : np.array, display_progress_bar : bool = False) -> None:
        self.reference = image
        self.palette = palette
        self.progress_bar = display_progress_bar

    def denoise_mask(self, mask : Image, refinement_steps : int = 7) -> np.array:
        pil_mask = Image.fromarray((mask[:, :, 0] + 1).astype(np.uint8))
        for _ in tqdm(range(refinement_steps), ncols=60, disable=not self.progress_bar, bar_format="|{bar}|{desc}: {percentage:3.0f}%"):
            pil_mask = pil_mask.filter(ImageFilter.ModeFilter)
        np_mask = np.array(pil_mask).astype(int) - 1
        return np_mask[:, :, None]
    
    def get_mask(self, denoising_steps : int = 7) -> np.array:
        color_layers = np.tile(self.palette[None, None, ...], (self.reference.shape[0], self.reference.shape[1], 1, 1))
        image_layers = np.tile(self.reference[:, :, None, 0:3], (1, 1, self.palette.shape[0], 1))
        mask = np.argmin(np.sum((color_layers - image_layers)**2, axis=-1), axis=-1, keepdims=True)
        mask[self.reference[:, :, 3][..., None] < 0.3] = -1
        if denoising_steps > 0:
            mask = self.denoise_mask(mask, refinement_steps=denoising_steps)
        return mask