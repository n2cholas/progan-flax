# Progressive Growing of GANs in Flax

Flax (JAX) implementation of [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf).
This code is meant to a starting point you can fork for your own needs rather than a 100% accurate reimplementation.

Some curated samples below from the CelebA (not HQ) dataset.
They're not as good as the original paper.

<!-- TODO: insert examples -->

## Usage

1. Download and extract [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Install [JAX](https://github.com/google/jax/) (instructions vary by system).
2. Install the other dependencies (ideally in a pip environment) via `pip install -r requirements.txt`. Requires Python >=3.6.
3. Run the code via `python src/main.py data_dir=<celeba directory>`.

This was originally run on a TPUv3-8.
You will need to adjust the hyperparameters in `src/config.yaml` for your local system (e.g. set `distributed: False`, decrease batch size, etc).

## Differences from Original Paper

- Different learning rates and batch sizes.
- Transition (interpolation between previous and current stage) only lasts 80% of each stage instead of entire stage.
- Slightly smaller model (with same architecture) since this implementation is for CelebA up to 128x128.
- Trained with `bfloat16` without loss scaling (as opposed to `float16` with loss scaling).
- tanh activation for the Generator outputs.
- Gain used for equalized learning rate adjusted for each activation instead of using `sqrt(2)` throughout (gains computed based on [PyTorch](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain)).
