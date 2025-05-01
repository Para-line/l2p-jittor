# L2P Jittor Implementation  
This repository contains a Jittor implementation of the "Learning to Prompt for Continual Learning" (L2P) paper.

## Prerequisites

### Obtaining Pre-trained Parameters
The pre-trained model parameters for this project are large and managed using **Git Large File Storage (LFS)**.

#### Method 1: Using Git LFS:

To ensure you download these large files correctly when cloning the repository, you should first install and set up Git LFS.

**For Ubuntu/Debian systems:** Open your terminal and run the following commands:
```
sudo apt install git-lfs
git lfs install # This command installs the necessary Git hooks.
```

Once Git LFS is installed and initialized, you can clone the repository as usual.

#### Method 2: Manual Download

If you prefer not to install Git LFS, or if you encounter issues with it, you can download the parameter files manually:

1. Navigate to the parameter files (under `params/`) in the GitHub repository's web interface.
2. Download the parameter files.
3. Place the parameter files under the `params/` directory within your local copy of the project repository.

### Environment

The code was developed and tested using the following environment:

- Ubuntu 20.04.5 LTS
- NVIDIA GeForce RTX 4090
- Python 3.9.0

### Required Packages

Install the necessary packages using pip:

```
jittor==1.3.9.14 # numpy will be installed as a dependency of jittor
tensorboard==2.19.0
pillow==9.2.0
```

### Change TensorBoard Directory

You may need to modify the `tensorboard_dir` argument at `configs/cifar100_l2p.py`

```
subparsers.add_argument('--tensorboard_dir', default='/root/tf-logs/', help='directory where TensorBoard logs will be saved.')
```

## Usage
### Training

```
python main.py \
       cifar100_l2p \
       --model vit_base_patch16_224 \
       --batch-size 16 \
       --data-path /local_datasets/ \
       --output_dir ./output
```

### Evaluation

```
python main.py cifar100_l2p --eval
```
## Result  
### Split-CIFAR100


| Metric                       | PyTorch | Jittor  |
| :--------------------------- |:--------|:--------|
| Average Forgetting           | 6.59%   | 6.56%   |
| Average Incremental Accuracy | 89.58%  | 89.29%  |
| Average Final Accuracy       | 83.82%  | 83.39%  |
| Total Training Time          | 0:20:40 | 0:29:36 |
| Average Epoch Time (sec)     | 22.18s  | 31.09s  |

| Metric                       | Description                                                                                                                                    |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| Average Incremental Accuracy | The average accuracy evaluated on each task *immediately after* that task has been trained.                                                    |
| Average Final Accuracy       | The average accuracy across all tasks evaluated *after* the model has finished training on the very *last* task.                               |
| Average Forgetting           | The average decrease in accuracy for each task compared between its highest accuracy and its accuracy *after* the final task has been trained. |


#### Final Accuracy Matrix

Pytorch:

```
    [97.9, 93.4, 91.4, 89.8, 88.6, 86.6, 87.1, 86.1, 84.3, 85.0], 
    [ 0.0, 94.9, 91.9, 88.7, 87.3, 85.4, 83.6, 84.2, 82.4, 81.2], 
    [ 0.0,  0.0, 90.7, 89.6, 85.6, 85.0, 83.2, 82.9, 82.1, 82.9], 
    [ 0.0,  0.0,  0.0, 89.6, 88.2, 86.9, 86.8, 85.2, 85.1, 84.7], 
    [ 0.0,  0.0,  0.0,  0.0, 90.9, 92.6, 90.4, 90.2, 88.6, 88.1], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0, 81.3, 80.4, 79.0, 79.5, 78.4], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 88.1, 83.4, 83.6, 81.0], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 84.6, 83.6, 81.0], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 89.9, 88.0], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 87.9]
```

Jittor :

```
    [98.5, 95.7, 93.5, 92.7, 92.0, 91.1, 89.6, 88.9, 87.1, 87.5], 
    [ 0.0, 94.7, 93.1, 89.9, 88.1, 85.8, 85.0, 84.3, 83.9, 82.7], 
    [ 0.0,  0.0, 89.0, 88.6, 86.6, 86.3, 85.2, 84.1, 83.6, 83.0], 
    [ 0.0,  0.0,  0.0, 88.9, 88.3, 84.6, 83.3, 83.4, 81.3, 78.9], 
    [ 0.0,  0.0,  0.0,  0.0, 90.8, 89.9, 89.5, 89.1, 87.4, 86.9], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0, 82.5, 81.3, 79.8, 80.8, 77.7], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 84.5, 82.4, 81.5, 80.3], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 84.6, 82.6, 80.0], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 93.1, 90.6], 
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 86.3]
```

**Note:** Detailed training logs, including metrics per epoch, can be visualized using TensorBoard by pointing it to the `tensorboard-logs` directory.

## Known Issues and Observations
During development and testing, the following issues related to the Jittor framework were observed:

### Issue 1: Platform-Specific Behavior of `index_fill_`

*   **File:** `issues/issue1.py`
*   **Description:** The operation `jt.index_fill_` fails to produce the expected results when run on a Windows operating system utilizing an AMD CPU.
*   **Note:** Correct behavior was observed when the same code was executed on the server.

### Issue 2: Numerical Precision Issues in Summation

*   **File:** `issues/issue2.py`
*   **Description:** Numerical precision issues observed in tensor summation. The framework's methods x.sum() and jt.sum(x) produce results that differ both from each other and from the sum calculated via a standard Python loop, with deviations reaching up to 1e-5 in both comparisons.
*   **Control Case:** `issue2_control.py` provides a comparison using PyTorch, where similar summation operations do not show this precision degradation.

### Related Issue: Precision and Reproducibility with `jt.set_global_seed`

*   **Description:** Setting the global random seed using `jt.set_global_seed` does not guarantee perfect reproducibility also due to precision issues in the generated random numbers.
*   **Observation:** Errors up to 1e-5 have been noted in the generated values. This can lead to significant divergence in outcomes over multiple runs.
