<div align="center">

# Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks

![Venue:ECCV 2024](https://img.shields.io/badge/Venue-ECCV%202024%20-blue)
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2301.11308&color=B31B1B)](https://arxiv.org/abs/2401.06187)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Abstract
Machine unlearning has become a pivotal task to erase the influence of data from a trained model. It adheres to recent data regulation standards and enhances the privacy and security of machine learning applications. In this work, we present a new machine unlearning approach Scissorhands. Initially, Scissorhands identifies the most pertinent parameters in the given model relative to the forgetting data via connection sensitivity. By reinitializing the most influential top-k percent of these parameters, a trimmed model for erasing the influence of the forgetting data is obtained. Subsequently, Scissorhands fine-tunes the trimmed model with a gradient projection-based approach, seeking parameters that preserve information on the remaining data while discarding information related to the forgetting data. Our experimental results, conducted across image classification and image generation tasks, demonstrate that Scissorhands, showcases competitive performance when compared to existing methods.

## Getting Started
The code is split into two subfolders, i.e., Classification and Stable Diffusion experiments. Detailed instructions are included in the respective subfolders.

## BibTeX
```
@article{wu2024scissorhands,
  title={Scissorhands: Scrub Data Influence via Connection Sensitivity in Networks},
  author={Wu, Jing and Harandi, Mehrtash},
  journal={arXiv preprint arXiv:2401.06187},
  year={2024}
}
```
## Acknowledgements
This repository makes liberal use of code from [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency), [Selective Amnesia](https://github.com/clear-nus/selective-amnesia) and [ESD](https://github.com/rohitgandikota/erasing/tree/main).
