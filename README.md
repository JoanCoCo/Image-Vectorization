# Image Vectorization
This repository provides a program based on Python to vectorize images. Therefore, it allows to process a given image to discretize its colors, use them to segment the image, generate vector shape layers and render them together to obtained an stylized copy of higher resolution.

![](https://github.com/JoanCoCo/Image-Vectorization/blob/main/images/examples.png?raw=true)

## Dependecies
- [PyTorch](https://pytorch.org)
- [NumPy](https://numpy.org)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [CarioSVG](https://cairosvg.org)
- [Potrace](https://potrace.sourceforge.net)

## Usage
```
main.py [-h] [--input INPUT] [--output OUTPUT] [--dpi DPI] [--palette-size PALETTE_SIZE] [--cell-size CELL_SIZE]
        [--grid-size GRID_SIZE] [--palette-warmup-steps PALETTE_WARMUP_STEPS]
        [--palette-refinement-steps PALETTE_REFINEMENT_STEPS] [--flattening-steps FLATTENING_STEPS]
        [--mask-denoising-steps MASK_DENOISING_STEPS] [--render-denoising-steps RENDER_DENOISING_STEPS]
        [--use-antialiasing] [--verbose VERBOSE] [--save-grid SAVE_GRID] [--save-palette SAVE_PALETTE]
        [--save-palette-weights SAVE_PALETTE_WEIGHTS] [--save-flat-image SAVE_FLAT_IMAGE] [--save-mask SAVE_MASK]
        [--save-layers SAVE_LAYERS] [--flat-image FLAT_IMAGE] [--palette PALETTE]
        [--palette-variety-weight PALETTE_VARIETY_WEIGHT] [--palette-coherence-weight PALETTE_COHERENCE_WEIGHT]
        [--potrace-path POTRACE_PATH] [--config CONFIG]

arguments:
  -h, --help                        Show this help message.
  --input INPUT                     Input image file (default: None).
  --output OUTPUT                   Filename for the the resulting image (default: None).
  --dpi DPI                         Level of detail used for rendering (default: 150).
  --palette-size PS                 Maximum number of colors used for the resulting image (default: 16).
  --cell-size CS                    Cell resolution used in the warm-up (default: 32x32).
  --grid-size GS                    Grid used in the warm-up (default: None).
  --palette-warmup-steps PWS        Number of steps used in the warm-up of the palette estimation (default: 501).
  --palette-refinement-steps PRS    Number of steps used in the palette estimation (default: 1001).
  --flattening-steps FS             Number of steps used in the estimation of an intermidiate flat image closer 
                                    to the palette (default: 0).
  --mask-denoising-steps MDS        Number of steps used for denoising the mask (default: 5).
  --render-denoising-steps RDS      Number of steps used for denoising the final image (default: 3).
  --use-antialiasing                Use antialiasing for the final image (default: False).
  --verbose VERBOSE                 Frequency of information updates given by the tool provided as
                                    "general_messages:palette_iterations:flattening_iterations" (default: 1:0:0).
  --save-grid SAVE_GRID             Saves the computed grid in the specified filename (default: None).
  --save-palette SAVE_PALETTE       Saves the computed palette in the specified filename (default: None).
  --save-palette-weights SPW        Saves the computed weights in the specified filename (default: None).
  --save-flat-image SFI             Saves the computed intermediate image in the specified filename (default: None).
  --save-mask SAVE_MASK             Saves the computed segmentation mask in the specified filename (default: None).
  --save-layers SAVE_LAYERS         Saves the computed vector layers in the specified folder (default: None).
  --flat-image FLAT_IMAGE           Load the flat image from a file (default: None).
  --palette PALETTE                 Load the palette from a file (default: None).
  --palette-variety-weight PVW      Use a custom weight for the palette variety loss (default: 1.0).
  --palette-coherence-weight PCW    Use a custom weight for the palette coherence loss (default: 1.0).
  --potrace-path POTRACE_PATH       Path to the local potrace installation (default: potrace).
  --config CONFIG                   Configuration file defining the arguments for the system (default: None).
```

Usage example: ``python main.py --input=data/zelda.jpg --output=zelda_hd/result.png --dpi=300 --palette-size=32 --cell-size=96x96 --flattening-steps=21``
