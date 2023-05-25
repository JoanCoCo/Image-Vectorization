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

def text_file_to_palette(filename : str) -> np.array:
    result = []
    use_float = False
    with open(filename, "r") as file:
        for line in file:
            use_float = use_float or ("type" in line and "float" in line and not "#" in line)
            if not "#" in line and not "type" in line:
                color = line.replace("\n", "").split(",")
                result.append([float(color[0]), float(color[1]), float(color[2])])
        file.close()
    if use_float:
        result = np.array(result).astype(float)
    else:
        result = np.array(result).astype(float) / 255.0
    return result

def palette_to_text_file(filename : str, palette : np.array, as_float : bool = False) -> None:
    upalette = palette
    with open(filename, "w") as file:
        file.write("# Palette encoded using RGB colors. #\n")
        if as_float:
            file.write("type float\n")
        else:
            file.write("type uint8\n")
            upalette = (palette * 255.0).astype(np.uint8)
        for color in upalette:
            if as_float:
                file.write("{:.5f},{:.5f},{:.5f}\n".format(color[0], color[1], color[2]))
            else:
                file.write("{:03d},{:03d},{:03d}\n".format(color[0], color[1], color[2]))
        file.close()

def display_elapsed_time(start : float, end : float, header : str = "Done in ") -> None:
    elapsed_palette = end - start
    if elapsed_palette / 60.0 < 1.0:
        print("{:s}{:d} sec.".format(header, int(elapsed_palette)))
    else:
        print("{:s}{:d} min.".format(header, int(elapsed_palette / 60.0)))

if __name__ == "__main__":
    ###############   ARGUMENTS   ###############
    parser = argparse.ArgumentParser(description="Python program to vectorize images. Given an input image, \
                                     it generates an upscaled version obtained from generating color vector \
                                     shape layers and stacking them together.",
                                     epilog="Usage example: python main.py --input=data/zelda/5.JPG --output=zelda_hd/result.png --dpi=300 --palette-size=32 --cell-size=96x96 --flattening-steps=21",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default=None, help="input image file")
    parser.add_argument("--output", type=str, default=None, help="filename for the the resulting image")
    parser.add_argument("--dpi", type=int, default=150, help="level of detail used for rendering")
    parser.add_argument("--palette-size", type=int, default=16, help="maximum number of colors used for the resulting image")
    parser.add_argument("--cell-size", type=str, default="32x32", help="cell resolution used in the warm-up")
    parser.add_argument("--grid-size", type=str, default=None, help="grid used in the warm-up")
    parser.add_argument("--palette-warmup-steps", type=int, default=501, help="number of steps used in the warm-up of the palette estimation")
    parser.add_argument("--palette-refinement-steps", type=int, default=1001, help="number of steps used in the palette estimation")
    parser.add_argument("--flattening-steps", type=int, default=0, help="number of steps used in the estimation of an intermidiate flat image closer to the palette")
    parser.add_argument("--mask-denoising-steps", type=int, default=5, help="number of steps used for denoising the mask")
    parser.add_argument("--render-denoising-steps", type=int, default=3, help="number of steps used for denoising the final image")
    parser.add_argument("--use-antialiasing", action='store_true', help="use antialiasing for the final image")
    parser.add_argument("--verbose", type=str, default="1:0:0", help="frequency of information updates given by the tool provided as \"general_messages:palette_iterations:flattening_iterations\"")
    parser.add_argument("--save-grid", type=str, default=None, help="saves the computed grid in the specified filename")
    parser.add_argument("--save-palette", type=str, default=None, help="saves the computed palette in the specified filename")
    parser.add_argument("--save-palette-weights", type=str, default=None, help="saves the computed weights in the specified filename")
    parser.add_argument("--save-flat-image", type=str, default=None, help="saves the computed intermediate image in the specified filename")
    parser.add_argument("--save-mask", type=str, default=None, help="saves the computed segmentation mask in the specified filename")
    parser.add_argument("--save-layers", type=str, default=None, help="saves the computed vector layers in the specified folder")
    parser.add_argument("--flat-image", type=str, default=None, help="load the flat image from a file")
    parser.add_argument("--palette", type=str, default=None, help="load the palette from a file")
    parser.add_argument("--palette-variety-weight", type=float, default=1.0, help="use a custom weight for the palette variety loss")
    parser.add_argument("--palette-coherence-weight", type=float, default=1.0, help="use a custom weight for the palette coherence loss")
    parser.add_argument("--potrace-path", type=str, default="potrace", help="path to the local potrace installation")
    parser.add_argument("--config", type=str, default=None, help="configuration file defining the arguments for the system")
    ARGS = parser.parse_args()
    if ARGS.config is not None:
        data = json.load(open(ARGS.config, 'r'))
        for key in data:
            ARGS.__dict__[key] = data[key]
    if ARGS.input is None or ARGS.output is None:
        print("ERROR: Both input and output files must be specified")
        exit()
    verbose_palette = int(ARGS.verbose.split(":")[1])
    verbose_flattening = int(ARGS.verbose.split(":")[2])
    verbose_general = int(ARGS.verbose.split(":")[0]) > 0
    cell_size = (int(ARGS.cell_size.split("x")[0]), int(ARGS.cell_size.split("x")[1]))
    grid_size = None
    if not ARGS.grid_size is None:
        grid_size = (int(ARGS.grid_size.split("x")[0]), int(ARGS.grid_size.split("x")[1]))
        cell_size = None
    verify_output_file_path(ARGS.output)
    #############################################

    start_time = time.time()

    ###############    PALETTE    ###############
    if ARGS.palette is None:
        if verbose_general:
            print("------------------------------------------------------------")
            print("Computing color palette...")
        start_palette = time.time()
        pg = PaletteGenerator_v2(ARGS.input, palette_size=ARGS.palette_size, cell_size=cell_size, grid_size=grid_size,
                                color_distance_weight=ARGS.palette_variety_weight, color_similarity_weight=ARGS.palette_coherence_weight,
                                display_progress_bar=verbose_general and not verbose_palette)
        if ARGS.palette_warmup_steps + ARGS.palette_refinement_steps > 0:
            pg.optimize_palette(iterations=(ARGS.palette_warmup_steps, ARGS.palette_refinement_steps), lr=0.01, verbose=verbose_palette)
        end_palette = time.time()
        if not ARGS.save_grid is None:
            verify_output_file_path(ARGS.save_grid)
            grid = pg.get_grid_palette(as_uint8=True, image_format=True)
            Image.fromarray(grid).save(ARGS.save_grid)
        palette = pg.get_palette()
        if not ARGS.save_palette is None:
            verify_output_file_path(ARGS.save_palette)
            if "png" in ARGS.save_palette or "jpg" in ARGS.save_palette or "jpeg" in ARGS.save_palette or "bmp" in ARGS.save_palette:
                palette_img = np.tile(palette[:, None, None, ...], (1, 128, 128, 1)).reshape(palette.shape[0] * 128, 128, 3)
                Image.fromarray((palette_img * 255.0).astype(np.uint8)).save(ARGS.save_palette)
            else:
                if ":f" in ARGS.save_palette:
                    palette_to_text_file(ARGS.save_palette[:-2], palette, as_float=True)
                else:
                    palette_to_text_file(ARGS.save_palette, palette, as_float=False)
        if not ARGS.save_palette_weights is None:
            verify_output_file_path(ARGS.save_palette_weights)
            Image.fromarray(pg.get_weights_image(as_uint8=True, with_color=False)).save(ARGS.save_palette_weights)
        if verbose_general:
            display_elapsed_time(start_palette, end_palette)
    else:
        palette = text_file_to_palette(ARGS.palette)
        if verbose_general:
            print("------------------------------------------------------------")
            print("Color palette loaded.")
    #############################################



    ###############  FLATTENING   ###############
    if ARGS.flat_image is None:
        if verbose_general:
            print("------------------------------------------------------------")
            print("Computing flattening...")
        start_flat = time.time()
        flt = Flattener(palette, ARGS.input, display_progress_bar=verbose_general and not verbose_flattening)
        if ARGS.flattening_steps > 0:
            flt.optimize(ARGS.flattening_steps, lr=0.01, verbose=verbose_flattening)
        end_flat = time.time()
        flat_image = flt.get_flat_image()
        if not ARGS.save_flat_image is None:
            verify_output_file_path(ARGS.save_flat_image)
            Image.fromarray((flat_image * 255.0).astype(np.uint8)).save(ARGS.save_flat_image)
        if verbose_general:
            display_elapsed_time(start_flat, end_flat)
    else:
        flat_image = np.array(Image.open(ARGS.flat_image).convert("RGBA")).astype(float) / 255.0
        if verbose_general:
            print("------------------------------------------------------------")
            print("Flat image loaded.")
    #############################################



    ###############     MASK      ###############
    if verbose_general:
        print("------------------------------------------------------------")
        print("Generating mask...")
    start_mask = time.time()
    mask_generator = LayerGenerator(flat_image, palette, display_progress_bar=False)
    final_mask = mask_generator.get_mask(denoising_steps=ARGS.mask_denoising_steps)
    end_mask = time.time()
    if not ARGS.save_mask is None:
        verify_output_file_path(ARGS.save_mask)
        Image.fromarray((((final_mask[:, :, 0] + 1) / palette.shape[0]) * 255.0).astype(np.uint8)).save(ARGS.save_mask)
    if verbose_general:
        display_elapsed_time(start_mask, end_mask)
    #############################################



    ##############  VECTORIZATION  ##############
    if verbose_general:
        print("------------------------------------------------------------")
        print("Generating vector layers...")
    start_layers = time.time()
    vectorizer = Vectorizer(palette, final_mask, ARGS.output, dpi=ARGS.dpi, 
                            layers_folder=ARGS.save_layers, potrace_src=ARGS.potrace_path,
                            remove_intermediate_files=ARGS.save_layers is None,
                            display_progress_bar=verbose_general)
    vectorizer.generate_vector_layers()
    end_layers = time.time()
    if verbose_general:
        display_elapsed_time(start_layers, end_layers)
        print("------------------------------------------------------------")
        print("Rendering final image...")
    start_render = time.time()
    vectorizer.render_vector_layers(refinement_steps=ARGS.render_denoising_steps, use_antialiasing=ARGS.use_antialiasing)
    end_render = time.time()
    if verbose_general:
        display_elapsed_time(start_render, end_render)
    #############################################

    end_time = time.time()
    if verbose_general:
        print("------------------------------------------------------------")
        display_elapsed_time(start_time, end_time, header="Finished in ")
        print("------------------------------------------------------------")
    