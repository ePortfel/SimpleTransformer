# SimpleTransformer
Prosty model transformera uzupełniającego tekst (LLM pretraining)

Model transformera do uczenia na tekstach, z użyciem biblioteki PyTorch.
Transformer działa na tokenach będących kawałkami wyrazów. Model wylicza prawdopodobieństwa wystąpienia kolejnego tokena na podstawie fragmentu tekstu już wygenerowanego lub podanego przez użytkownika.
Parametry modelu oraz procesu uczenia zdefiniowane w pliku config.py.

Dla Windows 11 i karty GeForce RTX polecam Python 3.12.7 oraz Cuda 12.4.1

Sposób użycia:

python trener.py model lalka

Wytrenuje model na pliku lalka.txt i zgra model do pliku lalka.mdl oraz tokenizację tekstu do pliku lalka.tkn. Jeśli w momencie wywołania plik modelu i tokenów istnieją, program wczyta je i będzie kontynuował trenowanie sieci.

python generator.py model lalka

Wczyta model z pliku lalka.mdl oraz tokenizację z pliku lalka.tkn. Następnie zapyta o początek tekstu, który ma dokończyć. W przypadku, gdy użytkownik nie poda niczego, model spróbuje zacząć samodzielnie.

Jeśli chcemy trenować model na własnych danych, które mamy w postaci listy plików txt, można je połączyć w jeden plik na którym pracuje transformer:

python konkatenator.py katalog mojkatalogplikowtxt plik mojplikdocelowy

poza połączeniem plików w jeden, konkatenator odrzuci pliki napisane w alfabetach innych niż łaciński.

Osadzanie tokenów w przestrzeni wektorowej dokonywane jest losowo i podlega uczeniu. Uproszczony obraz ułożenia tokenów względem siebie w przestrzeni można obejrzeć poleceniem:

python modelinfo.py model lalka
