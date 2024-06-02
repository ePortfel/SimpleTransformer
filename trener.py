import os
import sys
import torch

import transformer

if "dane" in sys.argv:
    plikdanych=sys.argv[sys.argv.index("dane")+1]
else:
    print("Brak pliku z danymi (parametr: dane)")
    exit(0)

plikmodelu=None
if "model" in sys.argv:
    plikmodelu=sys.argv[sys.argv.index("model")+1]
    print("Model będzie zapisany do pliku "+plikmodelu)
else:
    print("Brak wskazania pliku do zgrania modelu")
    exit(0)

pliktokenow=None
if "tokeny" in sys.argv:
    pliktokenow=sys.argv[sys.argv.index("tokeny")+1]
    print("Tokeny będą zapisane do pliku "+pliktokenow)
else:
    print("Brak wskazania pliku do zgrania tokenów")
    exit(0)

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Działamy na '+device)

transformer=transformer.Transformer(device)
if os.path.exists(pliktokenow) and os.path.exists(plikmodelu):
    model=transformer.trenuj(plikdanych,pliktokenow,plikmodelu)
else:
    model=transformer.trenuj(plikdanych)

print("Zapisuję tokeny do pliku "+pliktokenow)
transformer.tokenizer.zapisz_do_pliku(pliktokenow)

print("Zapisuję model do pliku "+plikmodelu)
torch.save(model.state_dict(),plikmodelu)