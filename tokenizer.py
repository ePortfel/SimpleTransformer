import config
from tokenizers import CharBPETokenizer
from tokenizers import Tokenizer

class BasicTokenizer:

    tokenizer=None

    def __init__(self):
        self.tokenizer=CharBPETokenizer()

    def generuj(self,plikd):
        self.tokenizer.train(files=[plikd],vocab_size=config.tokens)

    def wczytaj_z_pliku(self,plik):
        self.tokenizer=Tokenizer.from_file(plik)

    def koduj(self,tekst):
        return self.tokenizer.encode(tekst).ids

    def dekoduj(self,enc):
        return self.tokenizer.decode(enc)
    
    def slownik(self):
        return self.tokenizer.get_vocab()
    
    def zapisz_do_pliku(self,plik):
        self.tokenizer.save(plik)