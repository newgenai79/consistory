# ConsiStory: Training-Free Consistent Text-to-Image Generation [SIGGRAPH 2024]

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2402.03286)

### [[Project Website](https://research.nvidia.com/labs/par/consistory/)] [[Consistory NVIDIA NIM](https://build.nvidia.com/nvidia/consistory)]

> **ConsiStory: Training-Free Consistent Text-to-Image Generation**<br>
> Yoad Tewel<sup>1,2</sup>, Omri Kaduri<sup>3</sup>, Rinon Gal<sup>1,2</sup>, Yoni Kasten<sup>1</sup>, Lior Wolf<sup>2</sup>, Gal Chechik<sup>1</sup>, Yuval Atzmon<sup>1</sup> <br>
> <sup>1</sup>NVIDIA, <sup>2</sup>Tel Aviv University, <sup>3</sup>Independent

![](https://research.nvidia.com/labs/par/consistory/static/images/Teaser.jpg)

>**Abstract**: <br>
> Text-to-image models offer a new level of creative flexibility by allowing users to guide the image generation process through natural language. However, using these models to consistently portray the same subject across diverse prompts remains challenging. Existing approaches fine-tune the model to teach it new words that describe specific user-provided subjects or add image conditioning to the model. These methods require lengthy per-subject optimization or large-scale pre-training. Moreover, they struggle to align generated images with text prompts and face difficulties in portraying multiple subjects. Here, we present ConsiStory, a training-free approach that enables consistent subject generation by sharing the internal activations of the pretrained model. We introduce a subject-driven shared attention block and correspondence-based feature injection to promote subject consistency between images. Additionally, we develop strategies to encourage layout diversity while maintaining subject consistency. We compare ConsiStory to a range of baselines, and demonstrate state-of-the-art performance on subject consistency and text alignment, without requiring a single optimization step. Finally, ConsiStory can naturally extend to multi-subject scenarios, and even enable training-free personalization for common objects.

## Description
This repo contains the official code for our Consistory paper.

## TODO:
- [x] Release code!
- [ ] Add FLUX support

## Setup
To set up our environment, please run:
```bash
conda env create --file environment.yml
```

## Usage

### Run from Command Line
This command-line interface (CLI) allows you to generate batches of images with a consistent subject in different settings for each image prompt. The model offers two run modes: `batch` and `cached`.

#### Basic Usage
To generate images, run the following commands with the desired parameters:
```bash
python consistory_CLI.py --subject "a cute dog" --concept_token "dog" --settings "sitting in the beach" "in the circus" --out_dir "out"
```

#### Parameters

- **`--run_type`**: Defines the type of run. Options are:
  - `batch` (default): Generates images in a single batch based on the provided prompts.
  - `cached`: Generates anchor images with cached settings, allowing additional image generation based on these cached anchors.
- **`--gpu`**: Specifies the GPU device ID (default is `0`).
- **`--seed`**: Sets the random seed for consistent results (default is `40`).
- **`--mask_dropout`**: Defines the dropout rate for the extended attention (default is `0.5`).
- **`--same_latent`**: Boolean to indicate if the same noise should be used for each image (default is `False`).
- **`--style`**: Specifies the style or prefix for each prompt (default is `"A photo of "`).
- **`--subject`**: The main subject of the image, e.g., `"a cute dog"`.
- **`--concept_token`**: Tokens for the subjects to remain consistent, provided as a list for multiple consistent subjects (e.g. if you want both the dog and hat to remain consistent) (default is `["dog"]`).
- **`--settings`**: A list of settings or backgrounds for generating different scenarios, e.g., `"sitting in the beach"` or `"in the circus"`.
- **`--cache_cpu_offloading`**: Boolean to indicate whether to offload the anchors cache to the CPU. This reduces memory usage but may slow down generation.
- **`--out_dir`**: The output directory where generated images will be saved.

#### Example commands
1. **Batch generation:**
```bash
python consistory_CLI.py --subject "a cute dog" --concept_token "dog" --settings "sitting in the beach" "standing in the snow" "playing in the park" --out_dir "out"
```
This command generates a batch of images of "a cute dog" in three settings, "sitting in the beach", "standing in the snow" and "playing in the park", and saves the output in the out directory. Note that the first two settings will determine the subject's identity, but changing any subsequent settings will generate new images without altering the subject's appearance.

2. **Cached Anchor Generation:**
```bash
python consistory_CLI.py --run_type "cached" --subject "a cute dog" --concept_token "dog" --settings "sitting in the beach" "in the circus" "swimming in the sea" "standing on a boat" "in a pet food commercial" --out_dir "out"
```
This command uses the cached mode. It generates images of "a cute dog" in the first two settings ("sitting in the beach", "in the circus") as anchor images, caching them for efficiency. Additional images are then generated in the subsequent settings, "swimming in the sea", "standing on a boat", and "in a pet food commercial".

3. **Multiple Consistent Subjects:**
```bash
python consistory_CLI.py --subject "a cute dog" --concept_token "dog" "hat" --settings "wearing a hat" "standing in the snow" "wearing a hat, sitting in the park" --out_dir "out"
```
This command generates a batch of images with multiple consistent subjects, including both the dog and the hat.

### Run from Jupyter Notebook
Example usage in `consistory_notebook.ipynb`

## Tips and Tricks
- The model tends to perform better when the anchor prompts that define the subject's identity are simpler, while subsequent prompts can be more complex.
- To increase consistency, you can adjust the mask dropout value. Lower values will increase consistency but may reduce pose variation.

## Citation
If you make use of our work, please cite our paper:

```
@article{tewel2024training,
  title={Training-free consistent text-to-image generation},
  author={Tewel, Yoad and Kaduri, Omri and Gal, Rinon and Kasten, Yoni and Wolf, Lior and Chechik, Gal and Atzmon, Yuval},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={4},
  pages={1--18},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```