import config
from tokenizer import BasicTokenizer
import torch
import torch.nn as nn
from torch.nn import functional
from datetime import datetime

class KanalUwagi(nn.Module):

    def __init__(self,rozmiar_kanalu):
        super().__init__()
        self.key=nn.Linear(config.atrybuty,rozmiar_kanalu,bias=False)
        self.query=nn.Linear(config.atrybuty,rozmiar_kanalu,bias=False)
        self.value=nn.Linear(config.atrybuty,rozmiar_kanalu,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(config.rozmiar_bloku,config.rozmiar_bloku)))

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)                                           # (B,T,C)
        q=self.query(x)                                         # (B,T,C)
        wei=q@k.transpose(-2,-1)*C**-0.5                        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))  # (B, T, T)
        wei=functional.softmax(wei,dim=-1)                      # (B, T, T)
        v=self.value(x)                                         # (B,T,C)
        out=wei@v                                               # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class KanalyUwagi(nn.Module):

    def __init__(self,kanaly,rozmiar_kanalu):
        super().__init__()
        self.heads=nn.ModuleList([KanalUwagi(rozmiar_kanalu) for i in range(kanaly)])
        self.proj=nn.Linear(config.atrybuty,config.atrybuty)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return out

class FeedFoward(nn.Module):

    def __init__(self,atrb):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(atrb,4*atrb),nn.ReLU(),nn.Linear(4*atrb,atrb))

    def forward(self,x):
        return self.net(x)

class Blok(nn.Module):

    def __init__(self,atrybuty,kanaly):
        super().__init__()
        self.sa=KanalyUwagi(kanaly,int(atrybuty/kanaly))
        self.ffwd=FeedFoward(atrybuty)
        self.ln1=nn.LayerNorm(atrybuty)
        self.ln2=nn.LayerNorm(atrybuty)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x
    
class LLModel(nn.Module):

    device=None
    rozmiar_slownika=None

    def __init__(self,rozm,dev):
        super().__init__()
        self.device=dev
        self.rozmiar_slownika=rozm
        self.tablica_kodujaca_tokeny=nn.Embedding(self.rozmiar_slownika,config.atrybuty)
        self.tablica_kodujaca_pozycje=nn.Embedding(config.rozmiar_bloku,config.atrybuty)
        self.bloki=nn.Sequential(*[Blok(config.atrybuty,kanaly=config.kanaly_uwagi) for i in range(config.warstwy)])
        self.normalizator=nn.LayerNorm(config.atrybuty)
        self.linear=nn.Linear(config.atrybuty,self.rozmiar_slownika)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        kodowane_litery=self.tablica_kodujaca_tokeny(idx)
        kodowane_pozycje=self.tablica_kodujaca_pozycje(torch.arange(T,device=self.device))
        x=kodowane_litery+kodowane_pozycje
        x=self.bloki(x)
        x=self.normalizator(x)
        logits=self.linear(x)
        if targets is None:
            strata=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            strata=functional.cross_entropy(logits,targets)
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
        return x, y

    def wczytaj_model(self,pliktkn,plikmod):
        self.tokenizer=BasicTokenizer()
        self.tokenizer.wczytaj_z_pliku(pliktkn)
        self.rozmiar_slownika=len(self.tokenizer.slownik())
        self.koder=self.tokenizer.koduj
        self.dekoder=self.tokenizer.dekoduj
        model=LLModel(self.rozmiar_slownika,self.device)
        m=model.to(self.device)
        m.load_state_dict(torch.load(plikmod))
        return m

    def utworz_model(self,plikd):
        self.tokenizer=BasicTokenizer()
        self.tokenizer.generuj(plikd)
        self.rozmiar_slownika=len(self.tokenizer.slownik())
        self.koder=self.tokenizer.koduj
        self.dekoder=self.tokenizer.dekoduj
        model=LLModel(self.rozmiar_slownika,self.device)
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
            xb,yb=self.losuj_porcje()
            logits,strata=m(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            strata.backward()
            optimizer.step()
            if (krok%(config.iteracje/20)==0):
                print("Iteracja "+str(krok)+" z "+str(config.iteracje)+" strata "+str(round(strata.item(),4)))

        stoptime=datetime.now()
        print("Stop: "+stoptime.strftime("%H:%M:%S"))

        return m
