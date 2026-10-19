# AI-Based-Image-Restoration-Using-Generative-Deep-Learning

An educational project exploring how deep learning can restore degraded images. It includes a conditional GAN training experiment and a pretrained motion-deblurring workflow for testing uploaded images in Google Colab.

The project is organized as two related tracks: a small, reproducible synthetic experiment for learning model building, and an inference workflow using the pretrained [NAFNet](https://github.com/megvii-research/NAFNet) GoPro deblurring model. The GAN trained on the synthetic shapes is not the model used for real-photo inference.

## Project Overview

Image restoration estimates a clean image from a degraded observation. The degradation may include blur, noise, compression artifacts, or low resolution. This project focuses on blur and noise for the synthetic GAN experiment and motion deblurring for the pretrained NAFNet workflow.

The project demonstrates how to:

- Create paired degraded and clean training samples.
- Build a U-Net-style generator and a PatchGAN discriminator in PyTorch.
- Train a conditional GAN with adversarial and reconstruction losses.
- Apply a pretrained restoration model to an uploaded image.
- Compare the input, restored estimate, and clean reference when a target image is available.
- Evaluate paired outputs with mean absolute error (MAE) and peak signal-to-noise ratio (PSNR).

## Workflows

### 1. Synthetic Conditional GAN

The notebook generates 64 × 64 procedural images containing simple colored shapes. It creates a degraded counterpart by applying Gaussian blur and additive noise, then trains on 1,200 generated pairs.

The generator uses an encoder-decoder structure with skip connections. The PatchGAN discriminator classifies local regions using both the degraded input and a candidate clean image. Training combines binary cross-entropy adversarial loss with an L1 reconstruction loss weighted by `80.0`.

The notebook uses a fixed seed (`7`), a batch size of `32`, Adam optimizers, a learning rate of `0.0002`, and eight epochs by default. These settings make the example repeatable as a teaching exercise; they are not tuned for real photographs.

### 2. Pretrained Motion Deblurring

For real-image experiments, the notebook downloads the official NAFNet GoPro width-64 checkpoint and invokes its GoPro inference configuration. This avoids applying the shape-trained GAN to natural photos.

The upload step supports two modes:

1. **One clean image:** the notebook creates a synthetic motion-blurred version, runs restoration, and uses the original upload as the target.
2. **Two paired images:** upload the degraded input first and the matching clean target second. The notebook restores the degraded image and compares it with the supplied target.

Both modes display two comparisons:

- `Input | Restored | Target`
- `Degraded Input | Clean Target`

The clean target is a reference image supplied by the user or the original clean upload. A model cannot recover the true target image from a degraded input alone.

## Repository Structure

```text
.
├── AI-Based Image Restoration Using Generative Deep Learning.ipynb
├── colab_image_restoration_upload.py
├── conditional_gan_models.py
├── nafnet_degrade_and_restore.py
├── nafnet_paired_restoration_comparison.py
├── plot_training_losses.py
├── restoration_environment_setup.py
├── restoration_metrics.py
├── run_nafnet_restoration.py
├── save_restoration_model.py
├── setup_nafnet.py
├── synthetic_restoration_dataset.py
├── train_conditional_gan.py
└── visualize_restoration_results.py
```

## Getting Started

### Recommended: Google Colab

1. Upload the notebook to Google Colab and open it.
2. Select **Runtime → Change runtime type** and choose a GPU if available.
3. To run the synthetic GAN experiment, execute the notebook cells from top to bottom.
4. To run the pretrained NAFNet workflow, restart the runtime and run the final setup cell followed by the upload/inference cell. The setup downloads dependencies and the pretrained checkpoint.
5. Upload either one clean image or a degraded/clean pair as described above.

The setup scripts use `/content/NAFNet` and are intended for Google Colab. A CUDA GPU is optional; CPU inference may take longer.

### Local Environment for the Synthetic GAN

Create a Python environment and install the core dependencies:

```bash
python -m venv .venv
```

Activate the environment, then install packages:

```bash
python -m pip install torch numpy matplotlib pillow
```

For the NAFNet workflow, use Google Colab or follow the environment instructions in the [official NAFNet repository](https://github.com/megvii-research/NAFNet). Its dependencies and checkpoint are separate from the synthetic GAN requirements.

## Running the Synthetic GAN Components

The Python files are companion modules for the notebook and are intended to be kept together in one directory.

| File | Purpose |
| --- | --- |
| `restoration_environment_setup.py` | Seeds Python, NumPy, and PyTorch and selects CPU or CUDA. |
| `synthetic_restoration_dataset.py` | Generates clean procedural images, degraded pairs, a `DataLoader`, and a preview. |
| `conditional_gan_models.py` | Defines the generator, PatchGAN discriminator, and device. |
| `train_conditional_gan.py` | Trains the GAN and returns generator/discriminator loss history. |
| `plot_training_losses.py` | Plots the loss history. |
| `visualize_restoration_results.py` | Displays input, restored, and target samples. |
| `restoration_metrics.py` | Calculates MAE and PSNR for paired tensors. |
| `save_restoration_model.py` | Saves generator weights and run metadata. |

Run the training workflow from a notebook after creating `train_loader`, `generator`, and `discriminator`, or run the companion scripts from the same directory. The training script imports the dataset and model modules by filename.

## Results and Outputs

- `restoration_generator.pt` — generator weights, image size, and seed from the synthetic GAN workflow.
- `/content/degraded_input.png` — generated or uploaded degraded input in the Colab runtime.
- `/content/restored_image.png` — restored estimate in the Colab runtime.
- Loss plots, image comparisons, MAE, and PSNR — displayed by the notebook during execution.

Colab storage is temporary. Download output images and checkpoints you want to keep before the runtime is deleted.

## Evaluation

The notebook reports MAE and PSNR for paired synthetic examples. These metrics compare pixel values and do not fully measure perceptual quality. GAN training losses can fluctuate, so review the visual comparisons and test on held-out image pairs for meaningful evaluation.

For the NAFNet workflow, a supplied clean target enables direct visual comparison. Do not treat a restored estimate as ground truth: restoration models may smooth detail, leave blur, or introduce plausible-looking details.

## Limitations

- The synthetic training set contains simple shapes rather than natural photographs.
- The GAN is a small demonstration model and is not expected to restore real photographs.
- The pretrained NAFNet checkpoint was trained for GoPro-style motion deblurring. Performance may be limited on severe camera panning, unusual blur, noise, compression, scratches, or other restoration tasks.
- The generated one-image degradation is synthetic and may not match real camera motion.
- One-factor or single-seed experiments do not establish robust performance; run multiple seeds and evaluate on representative held-out data for research use.

## Future Improvements

- Train or fine-tune on representative aligned real-world image pairs.
- Add multiple degradation models, such as sensor noise, JPEG compression, and low-light artifacts.
- Compare the GAN with non-adversarial baselines and report PSNR, SSIM, and perceptual evaluations on held-out images.
- Add multi-seed runs and document runtime and hardware for reproducibility.
- Package inference behind a simple upload interface with downloadable outputs.

## Technologies

- Python and PyTorch
- NumPy and Pillow
- Matplotlib
- OpenCV for synthetic motion blur in the Colab workflow
- NAFNet pretrained motion-deblurring model

## References

- Chen et al., [NAFNet: Simple Baselines for Image Restoration](https://github.com/megvii-research/NAFNet), including the official GoPro model and inference configuration.
- PyTorch documentation: https://pytorch.org/docs/

## Author

**PRITHIYADARSHAN T**

Computer Science and Engineering (Artificial Intelligence & Machine Learning)

**GitHub:** [https://github.com/prithyadarshan](https://github.com/prithyadarshan)

**LinkedIn:** [https://www.linkedin.com/in/prithiyadarshan-thiyagarajan-379a4b2a3](https://www.linkedin.com/in/prithiyadarshan-thiyagarajan-379a4b2a3)
