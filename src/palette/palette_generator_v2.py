import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

class PaletteGenerator_v2():
    def __init__(self, src_image_file: str, palette_size : int, 
                 cell_size : tuple[int, int] = (32, 32), 
                 grid_size : tuple[int, int] = None,
                 color_distance_weight : float = 1.0,
                 color_similarity_weight : float = 1.0,
                 display_progress_bar : bool = False) -> None:
        self.progress_bar = display_progress_bar
        self.target = torch.tensor(np.array(Image.open(src_image_file).convert("RGBA"), dtype=float) / 255.0).float()
        self.image_resolution = (self.target.shape[0], self.target.shape[1])
        if not grid_size is None:
            self.grid_size = grid_size
            self.cell_resolution = (int(self.image_resolution[0] / grid_size[0]), int(self.image_resolution[1] / grid_size[1]))
        else:
            excess = (self.image_resolution[0] % cell_size[0], self.image_resolution[1] % cell_size[1])
            pad_width = (cell_size[0] - excess[0]) % cell_size[0]
            pad_height = (cell_size[1] - excess[1]) % cell_size[1]
            self.target = torch.permute(self.target, (2, 0, 1))
            self.target = torch.nn.functional.pad(self.target, (0, pad_height, 0, pad_width), mode="reflect")
            self.target = torch.permute(self.target, (1, 2, 0))
            self.image_resolution = (self.target.shape[0], self.target.shape[1])
            self.cell_resolution = cell_size
            self.grid_size = (int(self.image_resolution[0] / cell_size[0]), int(self.image_resolution[1] / cell_size[1]))
        # Initial color grid.
        self.grid_palette = self.generate_grid_by_means(full_size=False).clone().detach()
        self.grid_palette.requires_grad = True
        # Final palette estimation.
        self.palette_weights = torch.randn((palette_size, int(self.get_number_of_effective_cells()), 1), requires_grad=True)
        # Optimization weights.
        self.cdl_weight = color_distance_weight
        self.csl_weight = color_similarity_weight

    def get_general_palette(self) -> torch.Tensor:
        general_palette = self.grid_palette[self.get_grid_mask() > 0.0].reshape(-1, 3)
        #general_palette = torch.concat([general_palette, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])], dim=0)
        return general_palette

    def apply_grouping(self) -> torch.Tensor:
        weights = torch.tile(self.palette_weights, (1, 1, 3))
        general_palette = self.get_general_palette()
        general_palette = torch.tile(general_palette[None, ...], (self.palette_weights.shape[0], 1, 1))
        return torch.sum(general_palette * torch.nn.functional.softmax(weights, dim=1), dim=1)
    
    def get_weights_image(self, as_uint8=False, with_color=False) -> np.array:
        result = torch.zeros_like(self.grid_palette)
        weights = torch.tile(self.palette_weights, (1, 1, 3))
        norm_weights = torch.nn.functional.softmax(weights, dim=1)
        norm_weights = torch.sum(norm_weights, dim=0)
        norm_weights = torch.clip(norm_weights, 0.0, 1.0) #norm_weights / torch.max(norm_weights)
        result[self.get_grid_mask() > 0.0] = norm_weights.flatten()
        result = torch.tile(result, (1, 1, self.cell_resolution[0], self.cell_resolution[1], 1))
        result = torch.permute(result, (0, 2, 1, 3, 4))
        result = result.reshape(self.image_resolution[0], self.image_resolution[1], 3)
        if with_color:
            result = result * self.get_full_grid()
        if as_uint8:
            return (result.detach().numpy() * 255.0).astype(np.uint8)
        else:
            return result.detach().numpy()

    def generate_grid_by_means(self, full_size=True) -> torch.Tensor:
        referece = self.get_image(with_alpha=False)
        referece = referece.reshape((self.grid_size[0], self.cell_resolution[0], self.grid_size[1], self.cell_resolution[1], 3))
        referece = referece.permute((0, 2, 1, 3, 4))
        referece = torch.mean(referece, dim=(2, 3), keepdim=True)
        if not full_size:
            return referece
        grid = torch.tile(referece, (1, 1, self.cell_resolution[0], self.cell_resolution[1], 1))
        grid = torch.permute(grid, (0, 2, 1, 3, 4))
        return grid.reshape(self.image_resolution[0], self.image_resolution[1], 3)
    
    def get_full_grid(self) -> torch.Tensor:
        grid = torch.tile(self.grid_palette, (1, 1, self.cell_resolution[0], self.cell_resolution[1], 1))
        grid = torch.permute(grid, (0, 2, 1, 3, 4))
        return grid.reshape(self.image_resolution[0], self.image_resolution[1], 3)
    
    def get_image(self, with_alpha=False) -> torch.Tensor:
        if with_alpha:
            return self.target
        else:
            return self.target[:, :, 0:3]
    
    def get_mask(self) -> torch.Tensor:
        return torch.tile(self.target[:, :, 3][..., None], (1, 1, 3))
    
    def get_grid_mask(self, full_size=False) -> torch.Tensor:
        referece = self.get_mask()
        referece = referece.reshape((self.grid_size[0], self.cell_resolution[0], self.grid_size[1], self.cell_resolution[1], 3))
        referece = referece.permute((0, 2, 1, 3, 4))
        referece = torch.mean(referece, dim=(2, 3), keepdim=True).round()
        if not full_size:
            return referece
        grid = torch.tile(referece, (1, 1, self.cell_resolution[0], self.cell_resolution[1], 1))
        grid = torch.permute(grid, (0, 2, 1, 3, 4))
        return grid.reshape(self.image_resolution[0], self.image_resolution[1], 3)
    
    def get_number_of_effective_pixels(self):
        return torch.sum(self.get_mask())
    
    def get_number_of_effective_cells(self):
        return torch.sum(self.get_grid_mask(full_size=False).squeeze()[:, :, 0])

    def get_color_difference(self) -> torch.Tensor:
        color_plane = self.get_full_grid()
        true_mask = self.get_mask()
        image_plane = self.get_image(with_alpha=False)
        return torch.abs(color_plane * true_mask - image_plane * true_mask)
    
    def color_loss(self) -> torch.Tensor:
        return torch.sum(self.get_color_difference()) / self.get_number_of_effective_pixels()
    
    def color_distance_loss(self) -> torch.Tensor:
        palette = self.apply_grouping()[None, ...]
        sp = torch.squeeze(torch.tile(palette, (self.palette_weights.shape[0], 1, 1)))
        m = torch.norm(sp - torch.transpose(sp, 0, 1), dim=-1, p=2) / 1.7320508076  # sqrt(3) = 1.7320508076
        return torch.abs(1.0 - torch.mean(m))

    def color_similarity_loss_v1(self) -> torch.Tensor:
        reduced_palette = self.apply_grouping()
        general_palette = self.get_general_palette()
        val = torch.tensor(0.0)
        for i in range(reduced_palette.shape[0]):
            mono_palette = torch.tile(reduced_palette[i][None, ...], (general_palette.shape[0], 1))
            diff = torch.sum(torch.pow(general_palette - mono_palette, 2.0), dim=-1)
            val = val + torch.min(diff)
        return val
    
    def palette_similarity_score(self, a, b) -> torch.Tensor:
        val = torch.tensor(0.0)
        if len(a.shape) == 2:
            a_plus = torch.tile(a[:, None, ...], (1, b.shape[0], 1))
            b_plus = torch.tile(b[None, ...], (a.shape[0], 1, 1))
            diff = torch.sum(torch.pow(b_plus - a_plus, 2.0), dim=-1) / 3.0
            val = torch.sum(torch.min(diff, dim=-1)[0])
        elif len(a.shape) == 3:
            a_plus = torch.tile(a[:, :, None, ...], (1, 1, b.shape[0], 1))
            b_plus = torch.tile(b[:, None, ...], (1, a.shape[0], 1, 1))
            diff = torch.sum(torch.pow(b_plus - a_plus, 2.0), dim=-1) / 3.0
            val = torch.sum(torch.min(diff, dim=-1)[0], dim=-1)
        else:
            print("WARNING: Input tensors must be of dimension 2 or 3 for palette_similarity_score.")
        return val
    
    def color_similarity_loss_v2(self) -> torch.Tensor:
        reduced_palette = self.apply_grouping()
        general_palette = self.get_general_palette()
        #history_palette = general_palette[0][None, ...]
        ref_val = self.palette_similarity_score(general_palette, reduced_palette) / general_palette.shape[0]
        val = self.palette_similarity_score(general_palette[0][None, ...], reduced_palette)
        for i in range(1, general_palette.shape[0]):
            weight = self.palette_similarity_score(general_palette[i][None, ...], general_palette[0:i])
            if weight > ref_val:
                val = val + self.palette_similarity_score(general_palette[i][None, ...], reduced_palette)
            #history_palette = torch.concat([history_palette, general_palette[i][None, ...]], dim=0)
        return val
    
    def color_similarity_loss_v3(self) -> torch.Tensor:
        reduced_palette = self.apply_grouping()
        general_palette = self.get_general_palette()
        #history_palette = general_palette[0][None, ...]
        ref_val = self.palette_similarity_score(general_palette, reduced_palette) / general_palette.shape[0]
        weights = torch.full((general_palette.shape[0] - 1, general_palette.shape[0] - 1, 3), 100000)
        w_mask = torch.ones(general_palette.shape[0] - 1, general_palette.shape[0] - 1).tril()
        w_mask = torch.tile(w_mask[..., None], (1, 1, 3))
        history_palette = torch.tile(general_palette[None, ...], (general_palette.shape[0], 1, 1))[:-1,:-1]
        weights = (1.0 - w_mask) * weights + w_mask * history_palette
        general_batches = general_palette[1:, None, ...]
        weights = self.palette_similarity_score(general_batches, weights)
        weights = torch.concat([(ref_val * 10)[None, ...], weights])
        #weights = torch.tile(weights[..., None], (1, 3))
        relevant_general_palette = general_palette[weights > ref_val]
        val = self.palette_similarity_score(relevant_general_palette, reduced_palette)
        return val

    def optimize_palette(self, iterations : tuple[int, int] = 1000, lr : float = 0.01, verbose : int = 0) -> None:
        # Initialization of the grid.
        optimizer = torch.optim.Adam([self.grid_palette], lr=lr)
        for it in tqdm(range(iterations[0]), ncols=60, disable=not self.progress_bar, bar_format="|{bar}|{desc}: {percentage:3.0f}%"):
            optimizer.zero_grad()
            col = self.color_loss()
            loss = col
            if verbose > 0 and it % verbose == 0:
                print("(1) Iteration {:d}:".format(it))
                print("\tColor loss: {:.5f}".format(col.item()))
                print("\tTotal loss: {:.5f}".format(loss.item()))
            loss.backward()
            optimizer.step()
            self.grid_palette.data.clamp_(0.0, 1.0)
        # Refinement of the color palette.
        optimizer = torch.optim.Adam([self.palette_weights], lr=lr)
        for it in tqdm(range(iterations[1]), ncols=60, disable=not self.progress_bar, bar_format="|{bar}|{desc}: {percentage:3.0f}%"):
            optimizer.zero_grad()
            cdl = self.cdl_weight * self.color_distance_loss()
            csl = self.csl_weight * self.color_similarity_loss_v2()
            loss = cdl + csl
            if verbose > 0 and it % verbose == 0:
                print("(2) Iteration {:d}:".format(it))
                print("\tColor variety loss: {:.5f}".format(cdl.item()))
                print("\tColor coherence loss: {:.5f}".format(csl.item()))
                print("\tTotal loss: {:.5f}".format(loss.item()))
            loss.backward()
            optimizer.step()
            self.grid_palette.data.clamp_(0.0, 1.0)

    def get_grid_palette(self, as_uint8=False, image_format=False) -> np.array:
        palette = self.grid_palette.squeeze().reshape(-1, 3)
        if image_format:
            palette = self.get_full_grid()
        if as_uint8:
            return (palette.detach().numpy() * 255.0).astype(np.uint8)
        else:
            return palette.detach().numpy()
        
    def get_palette(self, as_uint8=False, sort=True) -> np.array:
        palette = self.apply_grouping()
        if sort:
            palette = torch.concat(sorted(palette, key=lambda x: 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2], reverse=True)).reshape(-1, 3)
        if as_uint8:
            return (palette.detach().numpy() * 255.0).astype(np.uint8)
        else:
            return palette.detach().numpy()