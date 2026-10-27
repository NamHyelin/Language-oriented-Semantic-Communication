
# Language-Oriented Semantic Communication
---
**Semantic Coding & Knowledge Distillation for Text-to-Image Generation**

This repository provides the implementation of a **language-oriented semantic communication framework for text-to-image generation**.

The framework investigates semantic coding, importance-aware word transmission, and knowledge distillation to improve the efficiency of language-based image generation over communication networks.

Rather than treating every word as equally important, the framework explores strategies for identifying and prioritizing influential words using language-model attention, image reconstruction quality, and linguistic structure.

The implementation includes multiple word-selection strategies, such as attention-based selection, LPIPS-based selection, semantic contribution matching, and structured semantic coding (SSC/SKD).

[![Python](https://img.shields.io/badge/Python-262139?style=for-the-badge&logo=python&logoColor=F472B6)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-252341?style=for-the-badge&logo=pytorch&logoColor=A78BFA)](#)
[![Semantic Communication](https://img.shields.io/badge/Semantic_Communication-202C49?style=for-the-badge&logoColor=79B8FF)](#)
[![Generative AI](https://img.shields.io/badge/Generative_AI-262139?style=for-the-badge&logoColor=F472B6)](#)

---

## Publication

**Language-Oriented Communication with Semantic Coding and Knowledge Distillation for Text-to-Image Generation**

**H. Nam**, J. Park, J. Choi, M. Bennis, and S.-L. Kim

*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Republic of Korea, 2024, pp. 13506–13510.*

**DOI:** [10.1109/ICASSP48485.2024.10446638](https://doi.org/10.1109/ICASSP48485.2024.10446638)

### Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{nam2024language,
  title={Language-Oriented Communication with Semantic Coding
         and Knowledge Distillation for Text-to-Image Generation},
  author={Nam, Hyelin and Park, Jihong and Choi, Jinho
          and Bennis, Mehdi and Kim, Seong-Lyun},
  booktitle={2024 IEEE International Conference on Acoustics,
             Speech and Signal Processing (ICASSP)},
  pages={13506--13510},
  year={2024},
  organization={IEEE},
  doi={10.1109/ICASSP48485.2024.10446638}
}
```

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-252341?style=for-the-badge&logo=ieee&logoColor=A78BFA)](https://doi.org/10.1109/ICASSP48485.2024.10446638)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-202C49?style=for-the-badge&logo=googlescholar&logoColor=79B8FF)](https://scholar.google.com/scholar?q=Language-Oriented+Communication+with+Semantic+Coding+and+Knowledge+Distillation+for+Text-to-Image+Generation)



---

## Getting Started


### System Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/51797ffe-d367-4426-8828-669ebf7e6511" width="95%" alt="Language-Oriented Semantic Communication Framework"/>
</p>

---


### 1. External Dependencies

First, download the following repositories.

**Detectron2 [1]**

```bash
git clone https://github.com/facebookresearch/detectron2.git
```

**Image2Paragraph [2]**

```bash
git clone https://github.com/showlab/Image2Paragraph.git
```

Configure the downloaded repositories and required dependencies according to your local environment.

### 2. Main Configuration

The primary method-selection options are configured in:

```text
main_flicker.py
```

Select the desired `method` and corresponding `importance` option to evaluate different semantic word-selection strategies.

---

## Methods and Importance Settings

The implementation supports multiple strategies for selecting the words transmitted from Alice to Bob.

### 1. Attention-Based and Model-Aware Selection

| Method | Importance | Description |
|---|---|---|
| `partBob` | `most_attentive` | Uses a language model from part of Bob's model to select the most influential or attentive word for transmission. |
| `partBob_clipTX` | — | Same as `partBob`, but assumes Bob has a CLIP language base. |
| `allBob` | `lowest LPIPS` | Simulates transmission using Bob's model and selects the word that yields the lowest LPIPS relative to the reference (desired) image. |
| `allBob_clipTX` | — | Same as `allBob`, but assumes Bob has a CLIP language base. |
| `Alice` | — | Uses a language model from part of Alice's model to select the most influential or attentive word for transmission. |
| `random` | — | Randomly transmits words. |

### 2. Semantic Contribution and Attention Matching

| Method | Importance | Description |
|---|---|---|
| `bothAtt` | `contribute` | Measures contribution using the sum of each word's impact on other words (row and column sums in the language-model attention matrix). Transmits the word whose contribution values are most similar between Alice and Bob, minimizing their difference. |
| `bothAtt` | `combination` | Measures the attention relationship between one word and another word. Transmits the word whose combination value with the previously transmitted word is most similar between Alice and Bob. The first word is selected using the `contribute` method. |
| `bothcrossatt` | — | Compares Alice's and Bob's contribution maps, which represent the image regions described or generated by each word. Transmits the word with the most similar contribution map. |

### 3. Linguistic-Priority Transmission

| Method | Importance | Description |
|---|---|---|
| `prompttune` | `nouns` | Transmits nouns first, following the original context sequence. |
| `prompttune` | `nounsverbs` | Transmits nouns and verbs first, following the context sequence. |
| `prompttune` | `heads` | **SSC:** Transmits head words first, following the context sequence. |
| `prompttune` | `heads_partBob` | Transmits head words first, prioritizing them using a language model from Bob's model, similarly to `partBob`. |
| `prompttune` | `heads_random` | Transmits head words first in a random sequence. |
| `prompttune` | `heads_skd` | **SKD:** Fine-tunes Alice and transmits head words first, following the context sequence. |

---

## Key Concepts

### Importance-Aware Word Transmission

Different words contribute differently to the generated image.

The implemented strategies explore word importance using:

- **Language-model attention:** Identifying influential words from the attention structure.
- **Image reconstruction quality:** Selecting words based on LPIPS with respect to the desired image.
- **Semantic contribution:** Measuring the impact of words on other words.
- **Cross-attention similarity:** Comparing image-region contribution maps between Alice and Bob.
- **Linguistic structures:** Prioritizing nouns, verbs, and head words.

### Structured Semantic Coding (SSC)

The `heads` option transmits linguistic head words first while preserving their contextual sequence.

Additional variants investigate how word ordering and priority selection affect transmission.

### Semantic Knowledge Distillation (SKD)

The `heads_skd` option fine-tunes Alice and transmits head words first in contextual order.

The approach investigates how knowledge distillation can support language-oriented semantic coding for text-to-image generation.

---

## Repository Configuration

The key entry point described in this repository is:

```text
main_flicker.py
```

The implementation compares different semantic transmission methods through its `method` and `importance` configurations.

**Method categories**

```text
Word Selection Methods
│
├── Language Model-Based
│   ├── partBob
│   ├── partBob_clipTX
│   ├── Alice
│   └── random
│
├── Reconstruction-Based
│   ├── allBob
│   └── allBob_clipTX
│
├── Attention Matching
│   ├── bothAtt
│   │   ├── contribute
│   │   └── combination
│   └── bothcrossatt
│
└── Linguistic-Priority Coding
    └── prompttune
        ├── nouns
        ├── nounsverbs
        ├── heads
        ├── heads_partBob
        ├── heads_random
        └── heads_skd
```



---

## References and Acknowledgments

This implementation makes use of existing research tools and frameworks.

**[1] Detectron2**

Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo, and R. Girshick.

GitHub: https://github.com/facebookresearch/detectron2

**[2] Image2Paragraph**

GitHub: https://github.com/showlab/Image2Paragraph

**[3] DAAM: Interpreting Stable Diffusion Using Cross Attention**

R. Tang, L. Liu, A. Pandey, Z. Jiang, G. Yang, K. Kumar, P. Stenetorp, J. Lin, and F. Ture.

*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023.*

Paper: https://aclanthology.org/2023.acl-long.310/

---

<p align="center">
  <sub>Developed for research on semantic communication and generative AI-based wireless image transmission.</sub>
</p>
