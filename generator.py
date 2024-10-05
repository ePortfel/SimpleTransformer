import sys
import torch

import config
import transformer

pliktokenow=None
plikmodelu=None

if "model" in sys.argv:
    nazwa=sys.argv[sys.argv.index("model")+1]
    pliktokenow=nazwa+".tkn"
    plikmodelu=nazwa+".mdl"
else:
    print("Brak wskazania modelu")
    exit(0)

device='cuda' if torch.cuda.is_available() else 'cpu'
print('Działamy na '+device)

transformer=transformer.Transformer(device)
model=transformer.wczytaj_model(pliktokenow,plikmodelu)
print("Liczba parametrów modelu ",round(sum(p.numel() for p in model.parameters())/1e6,2), "milionów")

sstart=input("Prompt: ")
cstart=torch.zeros((1,1),dtype=torch.long,device=device)
if sstart:
    cstart=torch.tensor([transformer.koder(sstart)],dtype=torch.long,device=device)

print(transformer.dekoder(model.pisz(idx=cstart,dlugosc=config.dlugosc_odpowiedzi)[0].tolist()))
