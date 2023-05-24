import os
import numpy as np
import subprocess
from PIL import Image
from PIL import ImageFilter
import cairosvg
import shutil

class Vectorizer():
    def __init__(self, palette : np.array, mask : np.array, output : str,
                 dpi : int = 150,
                 layers_folder : str = None,
                 potrace_src : str = "potrace-1.16.mac-x86_64/potrace",
                 remove_intermediate_files : bool = False) -> None:
        self.palette = palette
        self.histogram = []
        self.dpi = dpi
        self.potrace = potrace_src
        self.output = output
        self.remove_files = remove_intermediate_files
        for i in range(palette.shape[0]):
            v = np.count_nonzero(mask == i)
            self.histogram.append((i, v))
        self.histogram.sort(key=lambda x: x[1])

        self.palette = (palette * 255.0).astype(np.uint8)
        self.layer_masks = np.zeros((self.palette.shape[0], mask.shape[0], mask.shape[1], 1), dtype=int)
        accum = np.zeros_like(self.layer_masks[0])
        for i, _ in self.histogram:
            self.layer_masks[i] = (accum + np.where(mask == i, np.ones_like(mask), np.zeros_like(mask))).clip(0, 1)
            accum = self.layer_masks[i]
        self.layer_masks = 1 - self.layer_masks
        
        self.output_folder = os.path.dirname(self.output)
        if layers_folder is None:
            self.bmp_folder = os.path.join(self.output_folder, "layers", "bmp")
            self.svg_folder = os.path.join(self.output_folder, "layers", "svg")
            self.render_folder = os.path.join(self.output_folder, "layers", "render")
        else:
            self.bmp_folder = os.path.join(layers_folder, "bmp")
            self.svg_folder = os.path.join(layers_folder, "svg")
            self.render_folder = os.path.join(layers_folder, "render")
        if not os.path.exists(self.bmp_folder):
            os.makedirs(self.bmp_folder)
        if not os.path.exists(self.svg_folder):
            os.makedirs(self.svg_folder)
        if not os.path.exists(self.render_folder):
            os.makedirs(self.render_folder)
    
    def generate_vector_layers(self) -> None:
        n = self.palette.shape[0]
        for i, _ in self.histogram:
            fillcolor = np.squeeze(self.palette[i])
            fillcolor_hex = "{:02X}{:02X}{:02X}".format(fillcolor[0], fillcolor[1], fillcolor[2])
            bmp_filename = os.path.join(self.bmp_folder, "layer_{:d}.bmp".format(n))
            svg_filename = os.path.join(self.svg_folder, "layer_{:d}.svg".format(n))
            Image.fromarray((np.tile(self.layer_masks[i], (1, 1, 3)) * 255.0).astype(np.uint8)).save(bmp_filename)
            args = [self.potrace, "--svg", 
                    #"--width={:d}".format(self.width), 
                    #"--height={:d}".format(self.height), 
                    "--color=#{:s}".format(fillcolor_hex), 
                    bmp_filename,
                    "--output={:s}".format(svg_filename)]
            subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            n = n - 1

    def render_vector_layers(self, refinement_steps : int = 3) -> None:
        n = self.palette.shape[0]
        result = None
        for i, _ in self.histogram:
            svg_filename = os.path.join(self.svg_folder, "layer_{:d}.svg".format(n))
            render_filename = os.path.join(self.render_folder, "layer_{:d}.png".format(n))
            cairosvg.svg2png(url=svg_filename, write_to=render_filename, dpi=self.dpi)
            if result is None:
                result = Image.open(render_filename).convert("RGBA")
            else:
                result = Image.alpha_composite(Image.open(render_filename).convert("RGBA"), result)
            n = n - 1
        #result.save(os.path.join(self.output_folder, "result.png"))
        for _ in range(refinement_steps):
            result = result.filter(ImageFilter.ModeFilter)
        result.save(self.output)
        if self.remove_files:
            shutil.rmtree(self.bmp_folder)
            shutil.rmtree(self.svg_folder)
            shutil.rmtree(self.render_folder)
            if os.path.exists(os.path.join(self.output_folder, "layers")):
                shutil.rmtree(os.path.join(self.output_folder, "layers"))