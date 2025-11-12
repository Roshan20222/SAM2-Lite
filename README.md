# SAM2-Lite: Real-Time Video Segmentation on Edge Devices

## Overview

SAM2-Lite is a comprehensive research project that brings state-of-the-art video object segmentation capabilities to edge devices through memory-aware knowledge distillation. This repository contains the complete research paper, implementation, documentation, and deployment tools for deploying efficient video segmentation models on resource-constrained hardware.

## Abstract

Getting state-of-the-art video segmentation models to run on edge devices like smartphones and drones remains a fundamental challenge in computer vision. While models like SAM2 deliver impressive results, they require powerful GPUs and consume substantial energy, making them impractical for resource-constrained devices. SAM2-Lite presents a family of lightweight video segmentation models designed specifically for real-time inference on edge hardware through memory-aware knowledge distillation.

Our approach is built on three key innovations:

1. **Memory-Aware Distillation**: Teaches the student model not just to match the teacher's outputs, but to replicate its temporal reasoning by matching attention distributions and memory readouts.

2. **Learned Memory Pruning**: Intelligently selects which frame features to retain within strict device memory budgets through a trainable gating network.

3. **Adaptive Inference**: Dynamically adjusts resolution and memory usage based on real-time device performance using PID control.

## Key Results

- **Accuracy**: 83.1 JF on DAVIS 2017 (96% of SAM2's performance)
- **Efficiency**: 6.5x faster inference and 12x lower energy consumption
- **Performance**: Real-time operation (20-35 ms/frame) on Jetson edge devices
- **Scalability**: Three model variants (Tiny, Small, Base) for different accuracy-efficiency trade-offs

## Directory Structure

```
.
├── README.md                           # This file
├── research_paper/                     # Research paper and documentation
│   ├── 16research1.tex                # Main LaTeX paper source
│   ├── paper.pdf                      # Compiled research paper
│   └── references.bib                 # Bibliography
├── src/                               # Source code implementation
│   ├── models/                        # Model architectures
│   │   ├── sam2_lite_tiny.py         # Tiny variant
│   │   ├── sam2_lite_small.py        # Small variant
│   │   └── sam2_lite_base.py         # Base variant
│   ├── training/                      # Training scripts
│   │   ├── distillation.py           # Knowledge distillation training
│   │   ├── memory_pruning.py         # Memory pruning training
│   │   └── adaptive_inference.py     # Adaptive inference system
│   └── utils/                         # Utility functions
│       ├── data_loader.py            # Data loading utilities
│       ├── metrics.py                # Evaluation metrics
│       └── visualization.py          # Visualization tools
├── deployment/                        # Deployment tools
│   ├── tensorrt/                     # TensorRT optimization
│   ├── onnx/                         # ONNX Runtime deployment
│   └── mobile/                       # Mobile deployment
├── data/                              # Datasets
│   ├── youtube_vos/                  # YouTube-VOS dataset
│   ├── davis_2017/                   # DAVIS 2017 dataset
│   └── bdd100k/                      # BDD100K dataset
├── figures/                           # Research figures and visualizations
│   ├── architecture.pdf              # Model architecture diagram
│   ├── memory_pruning_behavior.pdf  # Memory pruning visualization
│   └── energy_accuracy_pareto.pdf  # Performance metrics
└── website/                           # Research paper website
    ├── index.html                    # Main website
    ├── css/                          # Styling
    └── js/                           # Interactive elements
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Roshan20222/SAM2-Lite.git
cd SAM2-Lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Inference

```python
from src.models import SAM2LiteSmall
from src.utils import VideoProcessor

# Load model
model = SAM2LiteSmall(pretrained=True)

# Process video
processor = VideoProcessor(model)
results = processor.process_video('path/to/video.mp4')
```

### Training

```bash
# Train with knowledge distillation
python src/training/distillation.py \
    --teacher_model sam2_base \
    --student_model sam2_lite_small \
    --dataset youtube_vos \
    --batch_size 8 \
    --epochs 25
```

## Methodology

### Memory-Aware Knowledge Distillation

Unlike standard knowledge distillation that only matches output masks, our approach explicitly teaches the student model to replicate the teacher's temporal reasoning:

1. **Cross-Attention Distribution Matching**: Match attention distributions over past frames
2. **Memory Readout Feature Matching**: Match memory readout features capturing temporal information

### Learned Memory Pruning

The key challenge in edge deployment is strict memory constraints. We develop a trainable gating network that:

- Scores memory token importance based on visual features, motion magnitude, prediction uncertainty, and temporal distance
- Uses differentiable top-k selection to fit within device memory budgets (256-512 tokens)
- Learns to preserve informative frames while discarding redundant ones

### Adaptive Inference

To handle device heterogeneity and runtime fluctuations, we implement a PID controller that:

- Monitors frame processing time
- Dynamically adjusts input resolution (0.5-1.0x)
- Adjusts memory window size (2-8 frames)
- Maintains target frame rates across diverse hardware

## Experimental Results

### DAVIS 2017 Benchmark

| Model | JF Score | Parameters | FPS (V100) | Energy (Wh) |
|-------|----------|------------|------------|-------------|
| SAM2 (Teacher) | 86.5 | 615M | 4.8 | 35.6 |
| SAM2-Lite Base | 84.5 | 38.4M | 18.7 | 2.9 |
| SAM2-Lite Small | 83.1 | 12.8M | 31.2 | 1.3 |
| SAM2-Lite Tiny | 79.8 | 5.2M | 42.5 | 0.8 |

### Edge Device Performance

| Device | Jetson Nano | Jetson TX2 | Jetson Orin | iPhone 14 |
|--------|------------|-----------|------------|----------|
| SAM2-Lite Tiny (ms) | 48 | 24 | 11 | 19 |
| SAM2-Lite Small (ms) | 72 | 35 | 17 | 28 |
| Real-time Threshold (30 FPS) | 33ms | 33ms | 33ms | 33ms |

## Paper Sections

### 1. Introduction
Motivates the problem of deploying video segmentation on edge devices and discusses real-world applications including medical imaging, autonomous robotics, and environmental monitoring.

### 2. Related Work
Surveys existing approaches in:
- Video Object Segmentation (VOS)
- Segment Anything Models (SAM)
- Knowledge Distillation
- Adaptive Computation and Pruning
- Edge Deployment and Quantization

### 3. Method
Detailed description of:
- Student architecture with lightweight vision encoders
- Memory-aware distillation loss formulation
- Learned memory pruning mechanism
- Runtime-adaptive inference system

### 4. Experiments
- Dataset descriptions (YouTube-VOS, DAVIS 2017, BDD100K, MOSE)
- Training procedure and implementation details
- Main results on DAVIS 2017 benchmark
- Edge device evaluation
- Long-term stability analysis
- Ablation studies

### 5. Discussion
Analyzes why memory-aware distillation works, discusses limitations, and outlines future directions.

## Ablation Studies

Our comprehensive ablation studies demonstrate:

- Attention matching alone: -2.9 JF
- Readout matching alone: -1.7 JF
- Both matching components: -4.5 JF
- Learned pruning vs. FIFO: +3.3 JF improvement
- Training from scratch (no teacher): -6.6 JF

## Datasets Used

1. **YouTube-VOS**: 3,471 videos with 65 object categories
2. **DAVIS 2017**: 60 training sequences for validation
3. **BDD100K**: Driving videos for domain diversity
4. **MOSE**: Complex multi-object scenes with occlusions

## Implementation Details

### Training
- Framework: PyTorch 2.0
- Mixed Precision Training: Enabled
- Total Training Time: 36 hours on 4 A100 GPUs
- Batch Size: 8 clips per GPU
- Learning Rate: Exponential decay (10^-4 → 10^-5)

### Deployment
- Quantization: INT8 (most layers), FP16 (attention)
- Frameworks Supported: TensorRT, ONNX Runtime, Core ML
- Languages Supported: Python, C++

## Feature Highlights

✅ **Three Model Variants**: Tiny (5.2M params), Small (12.8M params), Base (38.4M params)
✅ **Real-time Performance**: 20-35 ms/frame on edge devices
✅ **Energy Efficient**: Up to 27x reduction in energy consumption
✅ **Adaptive Runtime**: Automatic adjustment to device capabilities
✅ **Privacy-Preserving**: On-device processing, no cloud dependency
✅ **Cross-Platform**: Android, iOS, Linux (Jetson), Windows
✅ **Well-Documented**: Complete source code and research paper

## Performance Benchmarks

### Accuracy vs. Efficiency Trade-off
SAM2-Lite models achieve the best balance:
- 96% of SAM2's accuracy with 48x fewer parameters
- 6.5x faster inference
- 12x lower energy consumption

### Long-Term Tracking Stability
- SAM2-Lite Small: 82-83 JF over 5-minute sequences (1-2 point drift)
- FIFO Baseline: 80 → 71 JF (10-point drop)
- Demonstrates superior memory management through learned pruning

## Use Cases

### Medical Imaging
On-device video segmentation for surgery guidance without cloud transmission.

### Environmental Monitoring
Drone-based monitoring with 20+ hours battery life per charge.

### Autonomous Robotics
Sub-100ms response times for real-time object tracking.

### Wearable Devices
AR applications with smooth visual effects on mobile phones.

## Limitations and Future Work

### Current Limitations
1. **Resolution**: 16×16 patches limit small object performance
2. **Fixed Memory Budget**: Cannot adapt to scene complexity
3. **Temporal Horizon**: May discard important early frames in long videos
4. **Domain Shift**: Performance degrades on out-of-distribution videos

### Future Directions
1. Hierarchical patching for multi-scale feature extraction
2. Predicted optimal memory budgets from scene analysis
3. Hierarchical memory architectures (detailed recent, compressed distant)
4. Hardware-software co-design for next-generation edge devices

## Citation

If you use SAM2-Lite in your research, please cite:

```bibtex
@article{pandey2024sam2lite,
  title={SAM2-Lite: Bringing Real-Time Video Segmentation to Edge Devices Through Memory-Aware Knowledge Distillation},
  author={Pandey, Roshan},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

We thank:
- Meta AI team for open-sourcing SAM2
- NVIDIA for providing Jetson development boards
- Nepal Academy of Science and Technology (NAST) for research support
- Anonymous reviewers for constructive feedback

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact

**Author**: Roshan Pandey
**Email**: pandeyroshan2021@outlook.com
**Affiliation**: Department of Computer Science, Tribhuvan University, Kathmandu, Nepal

## Repository Links

- **Paper**: [Full Research Paper](./research_paper/paper.pdf)
- **Website**: [Interactive Research Showcase](./website/index.html)
- **Issues**: Report bugs or suggest features
- **Discussions**: Join community discussions

---

**Last Updated**: November 2024
**Maintained By**: Roshan Pandey
