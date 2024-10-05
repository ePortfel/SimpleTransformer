import sys
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

minx=None
maxx=None
miny=None
maxy=None
embedding_matrix=model.osadzenie_tokenow.weight.detach().cpu().numpy()
pca = PCA()
reduced = pca.fit_transform(embedding_matrix)
plt.figure(figsize=(10, 10))
plt.scatter(reduced[:, 0], reduced[:, 1],color="#CCCCCC")
for i,label in enumerate(range(config.tokens)):
    label=transformer.dekoder([i])
    x=reduced[i,0]
    y=reduced[i,1]
    if (minx==None or x<minx): minx=x
    if (maxx==None or x>maxx): maxx=x
    if (miny==None or y<miny): miny=y
    if (maxy==None or y>maxy): maxy=y
    plt.annotate(label, (x,y))
print ("Zakres 2D: x od "+str(minx)+" do "+str(maxx)+" y od "+str(miny)+" y do "+str(maxy))
plt.show()
