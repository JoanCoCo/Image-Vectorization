import os
import time
import json
import torch
import numpy as np
from PIL import Image
torch.manual_seed(209)
np.random.seed = 209
import argparse

from src.palette.palette_generator_v2 import PaletteGenerator_v2
from src.vector.vectorizer import Vectorizer
from src.layering.layer_generator import LayerGenerator
from src.color.flattener import Flattener

def verify_output_file_path(path) -> None:
    dirs = os.path.dirname(path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--palette-size", type=int, default=16)
    parser.add_argument("--cell-size", type=str, default="32x32")
    parser.add_argument("--grid-size", type=str, default=None)
    parser.add_argument("--palette-warmup-steps", type=int, default=501)
    parser.add_argument("--palette-refinement-steps", type=int, default=1001)
    parser.add_argument("--flattening-steps", type=int, default=61)
    parser.add_argument("--mask-denoising-steps", type=int, default=5)
    parser.add_argument("--render-denoising-steps", type=int, default=3)
    parser.add_argument("--verbose", type=str, default="1:0:0")
    parser.add_argument("--save-grid", type=str, default=None)
    parser.add_argument("--save-palette", type=str, default=None)
    parser.add_argument("--save-palette-weights", type=str, default=None)
    parser.add_argument("--save-flat-image", type=str, default=None)
    parser.add_argument("--save-mask", type=str, default=None)
    parser.add_argument("--save-layers", type=str, default=None)
    parser.add_argument("--palette-variety-weight", type=float, default=1.0)
    parser.add_argument("--palette-coherence-weight", type=float, default=1.0)
    parser.add_argument("--potrace-path", type=str, default="potrace-1.16.mac-x86_64/potrace")
    parser.add_argument("--config", type=str, default=None)
    ARGS = parser.parse_args()
    if ARGS.config is not None:
        data = json.load(open(ARGS.config, 'r'))
        for key in data:
            ARGS.__dict__[key] = data[key]

    verbose_palette = int(ARGS.verbose.split(":")[1])
    verbose_flattening = int(ARGS.verbose.split(":")[2])
    verbose_general = int(ARGS.verbose.split(":")[0]) > 0
    cell_size = (int(ARGS.cell_size.split("x")[0]), int(ARGS.cell_size.split("x")[1]))
    grid_size = None
    if not ARGS.grid_size is None:
        grid_size = (int(ARGS.grid_size.split("x")[0]), int(ARGS.grid_size.split("x")[1]))
        cell_size = None
    verify_output_file_path(ARGS.output)

    start_time = time.time()
    if verbose_general:
        print("Computing color palette...")
    pg = PaletteGenerator_v2(ARGS.input, palette_size=ARGS.palette_size, cell_size=cell_size, grid_size=grid_size,
                             color_distance_weight=ARGS.palette_variety_weight, color_similarity_weight=ARGS.palette_coherence_weight)
    pg.optimize_palette(iterations=(ARGS.palette_warmup_steps, ARGS.palette_refinement_steps), lr=0.01, verbose=verbose_palette)
    if not ARGS.save_grid is None:
        verify_output_file_path(ARGS.save_grid)
        grid = pg.get_grid_palette(as_uint8=True, image_format=True)
        Image.fromarray(grid).save(ARGS.save_grid)
    palette = pg.get_palette()
    if not ARGS.save_palette is None:
        verify_output_file_path(ARGS.save_palette)
        palette_img = np.tile(palette[:, None, None, ...], (1, 128, 128, 1)).reshape(palette.shape[0] * 128, 128, 3)
        Image.fromarray((palette_img * 255.0).astype(np.uint8)).save(ARGS.save_palette)
    if not ARGS.save_palette_weights is None:
        verify_output_file_path(ARGS.save_palette_weights)
        Image.fromarray(pg.get_weights_image(as_uint8=True, with_color=False)).save(ARGS.save_palette_weights)

    if verbose_general:
        print("Computing flattening...")
    flt = Flattener(palette, ARGS.input)
    flt.optimize(ARGS.flattening_steps, lr=0.01, verbose=verbose_flattening)
    flat_image = flt.get_flat_image()
    if not ARGS.save_flat_image is None:
        verify_output_file_path(ARGS.save_flat_image)
        Image.fromarray((flat_image * 255.0).astype(np.uint8)).save(ARGS.save_flat_image)

    if verbose_general:
        print("Generating mask...")
    mask_generator = LayerGenerator(flat_image, palette)
    final_mask = mask_generator.get_mask(denoising_steps=ARGS.mask_denoising_steps)
    if not ARGS.save_mask is None:
        verify_output_file_path(ARGS.save_mask)
        Image.fromarray((((final_mask[:, :, 0] + 1) / palette.shape[0]) * 255.0).astype(np.uint8)).save(ARGS.save_mask)

    if verbose_general:
        print("Generating vector layers...")
    vectorizer = Vectorizer(palette, final_mask, ARGS.output, dpi=ARGS.dpi, 
                            layers_folder=ARGS.save_layers, potrace_src=ARGS.potrace_path,
                            remove_intermediate_files=ARGS.save_layers is None)
    vectorizer.generate_vector_layers()
    if verbose_general:
        print("Rendering final image...")
    vectorizer.render_vector_layers(refinement_steps=ARGS.render_denoising_steps)

    end_time = time.time()
    if verbose_general:
        print("Finished in {:d} min.".format(int((end_time - start_time) / 60)))
    