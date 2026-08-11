# ensayo-apolo
Ejercicio para el semillero de usar Apolo para entrenar una red

## In Apolo

To check the available modules:

```
module avail
```

To use miniconda and cuda:

```
module load miniconda3/25.11.1
module load cuda/12.9
```

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda env create -f nn-train-env.yml
conda activate nn-train-env
```
Make sure your system’s NVIDIA driver and CUDA toolkit are properly installed.
You can check your CUDA version with:

 ```
nvidia-smi
 ```

Example output: 

 ```
CUDA Version: 12.8
 ```

To confirm that PyTorch detects your GPU and CUDA correctly, run:

 ```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
 ```

Example output:

 ```
2.8.0+cu128 12.8 True NVIDIA RTX 2000 Ada Generation Laptop GPU
 ```

 2.8.0+cu128   →  PyTorch version 2.8.0 compiled with CUDA 12.8

12.8          →  CUDA runtime version recognized by PyTorch

True          →  GPU is available and correctly detected

NVIDIA RTX 2000 Ada Generation Laptop GPU  →  Your GPU model

To verify the packages installed in your `nn-train-env` conda environment, you can use the following command:

 ```
conda list -n nn-train-env
 ```