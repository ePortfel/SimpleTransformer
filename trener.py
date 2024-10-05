import os
import sys
import torch

import transformer

plikdanych=None
plikmodelu=None
pliktokenow=None
if "model" in sys.argv:
    nazwa=sys.argv[sys.argv.index("model")+1]
    plikdanych=nazwa+".txt"
    plikmodelu=nazwa+".mdl"
    pliktokenow=nazwa+".tkn"
    print("Model będzie zapisany do pliku "+plikmodelu)
    print("Tokeny będą zapisane do pliku "+pliktokenow)
else:
    print("Brak wskazania modelu")
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