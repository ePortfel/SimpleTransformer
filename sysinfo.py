import torch

print('Wersja PyTorch: '+torch.__version__)

if torch.cuda.is_available():
    print('CUDA jest dostępna')
else:
    print('Brak CUDA')
