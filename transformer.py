import config
import random
from tokenizer import BasicTokenizer
import torch
import torch.nn as nn
from torch.nn import functional
from datetime import datetime

class ModulUwagi(nn.Module):

    def __init__(self, rozmiar_modulu, maska=False):
        super().__init__()
        self.maska=maska
        self.K=rozmiar_modulu
        self.key=nn.Linear(config.atrybuty, rozmiar_modulu, bias=False)
        self.query=nn.Linear(config.atrybuty, rozmiar_modulu, bias=False)
        self.value=nn.Linear(config.atrybuty, rozmiar_modulu, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(config.rozmiar_bloku,config.rozmiar_bloku)))

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)                                               # (PORCJA,CZAS,MODUWAGI)
        q=self.query(x)                                             # (PORCJA,CZAS,MODUWAGI)
        wei=q@k.transpose(-2,-1)*self.K**-0.5                       # (PORCJA,CZAS,MODUWAGI) @ (PORCJA,MODUWAGI,CZAS) -> (PORCJA,CZAS,CZAS)
        if self.maska:
            wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))  # (PORCJA,CZAS,CZAS)
        wei=functional.softmax(wei,dim=-1)                          # (PORCJA,CZAS,CZAS)
        v=self.value(x)                                             # (PORCJA,CZAS,MODUWAGI)
        out=wei@v                                                   # (PORCJA,CZAS,CZAS) @ (PORCJA,CZAS,MODUWAGI) -> (PORCJA,CZAS,MODUWAGI)
        return out

class ModulyUwagi(nn.Module):

    def __init__(self, moduly_uwagi, rozmiar_modulu_uwagi):
        super().__init__()
        self.heads=nn.ModuleList([ModulUwagi(rozmiar_modulu_uwagi,True) for i in range(moduly_uwagi)])
        self.proj=nn.Linear(moduly_uwagi*rozmiar_modulu_uwagi,config.atrybuty)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads],dim=-1) # (PORCJA,CZAS,MODUWAGI*LICZBAMODULOW)
        out=self.proj(out)                               # (PORCJA,CZAS,ATRYBUTY)
        return out

class FeedFoward(nn.Module):

    def __init__(self,atrb):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(atrb,config.rozmiarff*atrb),nn.ReLU(),nn.Linear(config.rozmiarff*atrb,atrb))

    def forward(self,x):
        return self.net(x)

class Blok(nn.Module):

    def __init__(self, atrybuty, rozmiar):
        super().__init__()
        self.sa=ModulyUwagi(rozmiar, config.rozmiarmu)
        self.ffwd=FeedFoward(atrybuty)
        self.ln1=nn.LayerNorm(atrybuty)
        self.ln2=nn.LayerNorm(atrybuty)

    def forward(self,x):
        xn=self.ln1(x)
        x=x+self.sa(xn)
        xn=self.ln2(x)
        x=x+self.ffwd(xn)
        return x
    
class LLModel(nn.Module):

    transformer=None
    device=None
    rozmiar_slownika=None

    def __init__(self,rozm,dev,transformer):
        super().__init__()
        self.transformer=transformer
        self.device=dev
        self.rozmiar_slownika=rozm
        self.osadzenie_tokenow=nn.Embedding(self.rozmiar_slownika, config.atrybuty)
        self.osadzenie_pozycji=nn.Embedding(config.rozmiar_bloku, config.atrybuty)
        self.bloki=nn.Sequential(*[Blok(config.atrybuty, rozmiar=config.moduly_uwagi) for i in range(config.warstwy)])
        self.normalizator=nn.LayerNorm(config.atrybuty)
        self.linear=nn.Linear(config.atrybuty,self.rozmiar_slownika)

    def debugdump(self,tns,od,do):
        tokens = [self.transformer.dekoder([id]) for id in (tns.cpu().numpy())[od:do]]
        return "|".join(tokens)

    def forward(self,idx,targets=None,ddebug=False):
        B,T=idx.shape
        wektory_liter=self.osadzenie_tokenow(idx)
        wektory_pozycji=self.osadzenie_pozycji(torch.arange(T, device=self.device))
        x=wektory_liter+wektory_pozycji
        x=self.bloki(x)
        x=self.normalizator(x)
        logits=self.linear(x)
        if targets is None:
            strata=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C) # (PORCJA*CZAS,SLOWNIK)
            targets=targets.view(B*T) # (PORCJA*CZAS)
            strata=functional.cross_entropy(logits,targets)
            if ddebug:
                dumpl = 40
                sidx = random.randint(0, max(0, len(idx) - dumpl))
                idx=idx.view(B*T)
                print("  Wejście: " + self.debugdump(idx,sidx,sidx+dumpl))
                print("      Cel: " + self.debugdump(targets,sidx,sidx+dumpl))
                predicted_labels=torch.argmax(logits,dim=1)
                print("Predykcja: " + self.debugdump(predicted_labels,sidx,sidx+dumpl))
        return logits,strata

    def pisz(self,idx,dlugosc):
        for i in range(dlugosc):
            fragment=idx[:,-config.rozmiar_bloku:]
            logits,strata=self(fragment)
            logits=logits[:,-1,:]                           #(B,C)
            probs=functional.softmax(logits,dim=-1)         #(B,C)
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            idx=torch.cat((idx,idx_next),dim=1)             #(B,T+1)
        return idx

class Transformer:

    device=None
    dane=None
    koder=None
    dekoder=None
    rozmiar_slownika=None
    tokenizer=None

    def __init__(self,dev):
        self.device=dev

    def losuj_porcje(self):
        ix = torch.randint(len(self.dane) - config.rozmiar_bloku, (config.rozmiar_porcji,))
        x = torch.stack([self.dane[i:i+config.rozmiar_bloku] for i in ix])
        y = torch.stack([self.dane[i+1:i+config.rozmiar_bloku+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y     # x(PORCJA,CZAS) y(PORCJA,CZAS) y przesunięte o 1 jednostkę czasu do przodu

    def wczytaj_model(self,pliktkn,plikmod):
        self.tokenizer=BasicTokenizer()
        self.tokenizer.wczytaj_z_pliku(pliktkn)
        self.rozmiar_slownika=len(self.tokenizer.slownik())
        self.koder=self.tokenizer.koduj
        self.dekoder=self.tokenizer.dekoduj
        model=LLModel(self.rozmiar_slownika,self.device,self)
        m=model.to(self.device)
        m.load_state_dict(torch.load(plikmod))
        return m

    def utworz_model(self,plikd):
        self.tokenizer=BasicTokenizer()
        self.tokenizer.generuj(plikd)
        self.rozmiar_slownika=len(self.tokenizer.slownik())
        self.koder=self.tokenizer.koduj
        self.dekoder=self.tokenizer.dekoduj
        model=LLModel(self.rozmiar_slownika,self.device,self)
        m=model.to(self.device)
        return m

    def trenuj(self,plikd,pliktkn=None,plikmod=None):
        if pliktkn!=None and plikmod!=None:
            print("Wczytuje model w celu dalszego trenowania.")
            m=self.wczytaj_model(pliktkn,plikmod)
        else:
            print("Tworzę nowy model")
            m=self.utworz_model(plikd)
        with open(plikd,'r',encoding='utf-8') as f:
            tekst=f.read()
        self.dane=torch.tensor(self.koder(tekst),dtype=torch.long)
        optimizer=torch.optim.AdamW(m.parameters(),lr=config.stopa_optymalizacji)
        print("Liczba parametrów modelu ",round(sum(p.numel() for p in m.parameters())/1e6,2), "milionów")

        print("Start: "+datetime.now().strftime("%H:%M:%S"))

        for krok in range(config.iteracje):
            dumpkrok=krok%(config.iteracje/20)==0
            xb,yb=self.losuj_porcje()
            logits,strata=m(xb,yb,dumpkrok and config.debug)
            optimizer.zero_grad(set_to_none=True)
            strata.backward()
            optimizer.step()
            if (dumpkrok):
                print("Iteracja "+str(krok)+" z "+str(config.iteracje)+" strata "+str(round(strata.item(),4)))

        stoptime=datetime.now()
        print("Stop: "+stoptime.strftime("%H:%M:%S"))

        return m
