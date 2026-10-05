# Evaluation of Denoising Architectures on a Reconstructed Ultra-Low-Dose CT Dataset

Training and evaluation code for the ENIAC 2026 paper *Evaluation of Denoising
Architectures on a Reconstructed Ultra-Low-Dose CT Dataset* (Zago, Contassot,
Ravazio, Ughini, Kupssinskü and Barros; MALTA Lab, PUCRS and Kunumi Institute).

The paper compares ten low-dose CT denoising models on an ultra-low-dose CT
(ULDCT) dataset. The dataset was built from the AAPM-Mayo 2016 Low Dose CT Grand
Challenge by simulating 10% of the normal dose from the full-dose (NDCT) slices.
Every model is trained and tested on the same patient split, with the same
pre-processing and the same metrics. Hyperparameters follow each model's
original publication.

## Results (10% dose, test patient L506)

| Model     | Paradigm                          | SSIM ↑     | PSNR (dB) ↑ | RMSE (HU) ↓ |
|-----------|-----------------------------------|------------|-------------|-------------|
| U-Net\*   | CNN encoder-decoder               | 0.4869     | 11.2591     | 109.8445    |
| RED-CNN   | Residual CNN autoencoder          | 0.7946     | 24.2491     | 24.6240     |
| WGAN-VGG  | Wasserstein GAN                   | 0.7516     | 21.8028     | 32.5813     |
| EDCNN     | CNN with edge enhancement         | 0.7963     | 24.1284     | 24.9627     |
| CTformer  | Transformer                       | 0.7804     | 23.3289     | 27.3368     |
| DPN/FRN   | CNN, attention, dynamic conv.     | 0.8103     | 25.3777     | 21.6592     |
| CoreDiff  | Generalized diffusion             | **0.8385** | **27.4008** | **17.2163** |
| FGDM      | Frequency-guided diffusion GAN    | 0.6623     | 15.0843     | 70.8935     |
| SAD       | Schrödinger bridge diffusion      | 0.7803     | 23.6373     | 26.3983     |
| NEED      | Dual-domain diffusion             | 0.7436     | 21.3967     | 34.1332     |

\* The U-Net is an under-trained control: SGD with lr 1e-2, batch size 1 and 20
epochs, following its original configuration. Read its numbers as a lower
bound, not as evidence against the architecture (see Section 4 of the paper).

All values are means over the test slices. They come from a single test
patient and a single seed, so no variance is reported.

## Repository layout

```
common/          code shared by all models
  config.py      constants, command-line interface, logging
  data.py        file discovery, LD/FD pairing, normalisation, patching
  metrics.py     PSNR, SSIM, RMSE
  checkpoint.py  checkpoint save/load/resume
  evaluation.py  validation, test loop, figures, metric reports
  runtime.py     helpers for the script entry points
img2sino.py      fan-beam image <-> sinogram projector used by NEED
<model>.py       one script per model: architecture, training, testing
```

| Script        | Model    | Reference                 |
|---------------|----------|---------------------------|
| `unet.py`     | U-Net    | Ronneberger et al., 2015  |
| `red_cnn.py`  | RED-CNN  | Chen et al., 2017         |
| `wgan_vgg.py` | WGAN-VGG | Yang et al., 2018         |
| `edcnn.py`    | EDCNN    | Liang et al., 2020        |
| `ctformer.py` | CTformer | Wang et al., 2023         |
| `dpn.py`      | DPN/FRN  | Yang et al., 2023         |
| `corediff.py` | CoreDiff | Gao et al., 2023          |
| `fgdm.py`     | FGDM     | Li et al., 2023           |
| `sad_1.py`    | SAD      | Du et al., 2024           |
| `need.py`     | NEED     | Gao et al., 2025          |

## Dataset

### Source and split

The base data is the [AAPM-Mayo 2016 Low Dose CT Grand
Challenge](https://www.aapm.org/grandchallenge/lowdosect/). Ten of its patients
have both NDCT and 25%-dose LDCT. The quarter-dose LDCT is not used. The ULDCT
inputs are simulated from the NDCT slices instead.

| Split        | Patients                    |
|--------------|-----------------------------|
| `train`      | the remaining eight patients|
| `validation` | L143                        |
| `test`       | L506                        |

### ULDCT simulation

The simulation follows the fan-beam noise model of SAD (Du et al., 2024). It
was run with the ASTRA Toolbox on GPU. The reconstruction script is not part of
this repository.

| Parameter                | Value             |
|--------------------------|-------------------|
| Source-detector distance | 1270.0 mm         |
| Source-origin distance   | 870.0 mm          |
| Origin-detector distance | 400.0 mm          |
| Detectors                | 848               |
| Detector spacing         | 0.60 mm           |
| Views                    | 720               |
| Angular increment        | 0.5° (360° cover) |

For each NDCT slice:

1. HU is converted to attenuation, `mu = (HU / 1000 + 1) * mu_water`, with
   `mu_water = 0.02 mm^-1`.
2. A clean sinogram `p` is forward-projected from `mu`.
3. Photon counts are sampled as
   `N_i ~ Poisson(I0 * exp(-p_i)) + Normal(0, sigma_e^2)`, with `I0 = 2e4`
   (10% dose) and `sigma_e^2 = 10`.
4. The noisy sinogram is `p_i_noise = -ln(max(N_i, 1) / I0)`.
5. The image is reconstructed with FBP (Parzen filter) at 512x512 and native
   pixel spacing (~0.665 mm).
6. `mu` is converted back to HU, and the slice is stored in the
   [-1024, 3072] HU window.

### Expected layout

```
<data-root>/
  train/ validation/ test/
    images_low_dose/**/<patient>/*.npy                 ULDCT slices (HU)
    images_normal_dose/<patient>/full_1mm/*.IMA        NDCT targets (DICOM)
```

Patient ids follow the Mayo naming (`L067`, `L096`, ...). Inputs and targets
are paired by sorted slice position within each patient. If `validation` is
missing, per-epoch evaluation falls back to `test`. Both input and target are
normalised from [-1024, 3072] HU to [0, 1].

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, torch 2.14, torchvision 0.29, numpy 2.5, scipy 1.18,
matplotlib 3.11 and pydicom 3.0. Optional dependencies:

- `opencv-python-headless`: bilateral-filtered edge maps for FGDM. A NumPy
  fallback is used otherwise.
- `einops`: the DGDiff refinement stage of NEED.
- `torch-radon`: the fan-beam projector used by NEED's paper. A native PyTorch
  projector is used otherwise.

## Usage

Each script trains, keeps the checkpoints and then tests on the test split. The
paper results use `--dose 10pct`:

```bash
CUDA_VISIBLE_DEVICES=0 python corediff.py --dose 10pct --data-root /data/uldct_10pct/dataset
```

The scripts also accept `--dose 5pct`. The paper reports only the 10% setting.

### Common options

`python <model>.py --help` lists every option of a script.

| Option           | Default                                               |
|------------------|-------------------------------------------------------|
| `--dose`         | `5pct` (`5pct` or `10pct`)                            |
| `--data-root`    | `./data/uldct_<dose>/dataset`                         |
| `--output-dir`   | `./outputs/<dose>/<model>_<dose>`                     |
| `--num-workers`  | `8`                                                   |
| `--abdomen-idx`  | test slice saved as `result_abdomen.png`              |
| `--abdomen-frac` | `0.5` (relative position used when no index is given) |

### Model-specific options

| Script        | Options |
|---------------|---------|
| `corediff.py` | `--fd-cache-dir` |
| `ctformer.py` | `--fd-cache-dir` |
| `edcnn.py`    | `--torch-hub-dir` (ResNet-50 weights for the perceptual loss) |
| `fgdm.py`     | `--fd-cache-dir`, `--no-fd-cache`, `--no-compile` |
| `need.py`     | `--test-only`, `--fd-cache-dir`, `--no-fd-cache`, `--dgdiff-root`, `--no-dgdiff`, `--radon-backend`, `--require-torch-radon`, `--ray-samples`, `--angle-chunk` |
| `sad_1.py`    | `--pidinet-weights` (downloaded if missing), `--sample-steps` (1 = SAD-1, 5 = SAD-5) |

`--fd-cache-dir` stores the decoded DICOM targets as `.npy`, so `pydicom` is
taken out of the data-loading path. It does not change any value.

`wgan_vgg.py` and `edcnn.py` download the ImageNet VGG-19 and ResNet-50 weights
through torchvision on first use. NEED's DGDiff stage needs the
`denoising_diffusion_pytorch` package from NEED-main/DGDiff (`--dgdiff-root`)
and a trained `dgdiff_stage2.pt` in the output folder. Without them, only
SPDiff is run.

### Outputs

Each run writes the following to `--output-dir`:

- `<model>_<dose>_last.pt`: written every epoch. An interrupted run resumes
  from it.
- `<model>_<dose>_best.pt`: the checkpoint with the best validation PSNR.
- `losses_<dose>.npy`: the training losses.
- `metrics_<dose>.txt`: the test SSIM, PSNR and RMSE.
- `fig/`: test figures, including `result_abdomen.png`.

## Training setup

The hyperparameters follow each original publication (Table 3 of the paper).
For the diffusion models, the iteration count is the real stopping criterion.
Their epoch count is only a nominal reference.

| Model    | Learning rate | Batch | Optimizer         | Epochs         |
|----------|---------------|-------|-------------------|----------------|
| U-Net    | 1.0e-2        | 1     | SGD with momentum | 20             |
| RED-CNN  | 1.0e-4        | 16    | Adam              | 100            |
| WGAN-VGG | 1.0e-6        | 16    | Adam (G+D, betas) | 100            |
| EDCNN    | 1.0e-3        | 32    | AdamW             | 200            |
| CTformer | 1.0e-5        | 16    | Adam              | 4000           |
| DPN/FRN  | 1.0e-4        | 16    | Adam              | 500 (+100 FRN) |
| CoreDiff | 2.0e-4        | 4     | Adam              | 10000          |
| FGDM     | 1.0e-4        | 8     | Adam (G+D, betas) | 200            |
| SAD      | 5.0e-4        | 4     | Adam              | 10000          |
| NEED     | 1.0e-4        | 8     | Adam              | 10000          |

- Patch-based models train on 64x64 patches. At inference, each model works on
  the full slice or on overlapping patches, as in its original protocol.
- Augmentation (flips and rotations) is applied to the training set only. The
  same transform is applied to the input and its target.
- DPN/FRN pre-trains the FRN for 100 epochs before joint training.
- Learning-rate schedules from the original papers are kept. DPN uses decay,
  and CTformer decays the rate to 1e-6.
- The diffusion models use few-step sampling at inference: ten steps for
  CoreDiff and one to five for SAD.
- The experiments ran on two NVIDIA Tesla T4 GPUs (Kaggle) and an NVIDIA RTX
  A6000 (48 GB).

## Evaluation protocol

Metrics are computed in HU on the soft-tissue window [-160, 240] HU (data
range 400) over full 512x512 test slices, with these exceptions:

- `unet.py` evaluates the 388x388 centre crop produced by the unpadded U-Net.
- `fgdm.py` shifts the images by +160 HU before SSIM. PSNR and RMSE are not
  affected.
- `need.py` uses 3-slice context, so the first and last slice of each test
  patient are not evaluated.
- `unet.py`, `corediff.py` (EMA weights) and `fgdm.py` evaluate the weights of
  the final epoch instead of `<model>_<dose>_best.pt`.

## Citation

```bibtex
@inproceedings{zago2026uldct,
  title     = {Evaluation of Denoising Architectures on a Reconstructed
               Ultra-Low-Dose {CT} Dataset},
  author    = {Zago, Diogo L. and Contassot, Ra{\'i}ssa X. and Ravazio, Rafaela C.
               and Ughini, Augusto O. and Kupssinsk{\"u}, Lucas S.
               and Barros, Rodrigo C.},
  booktitle = {Anais do Encontro Nacional de Intelig{\^e}ncia Artificial e
               Computacional (ENIAC)},
  year      = {2026}
}
```

## Acknowledgments

This study was supported by the Center for Innovation and Artificial
Intelligence in Health (CI-IA Saúde), with partial funding from FAPESP (Grant
2020/09866-4), FAPEMIG (Grant PPE-00030-21), UNIMED Belo Horizonte, CAPES
(Finance Code 001), CNPq (Grant 443072/2024-8) and Kunumi Institute.
