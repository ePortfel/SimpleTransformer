# Parametry modelu o 140mln parametrów  (GPT-3 ma 175mld parametrów)

# Parametry modelu
tokens=200                  # liczba tokenów w słowniku
rozmiar_bloku=160           # Rozmiar fragmentu tekstu (w tokenach) do analizy kolejnego tokena (GPT-3: 2048)
moduly_uwagi=9              # Liczba równoległych kanałów aplikacji mechanizmu uwagi (GPT-3: 96)
warstwy=2                   # Liczba warstw przetwarzania (GPT-3: 96)
atrybuty=2000               # Liczba cech kodujących token tekstu (GPT-3: 12288)
rozmiarmu=100               # Rozmiar warstw wewnętrznych modułu uwagi
rozmiarff=8                 # Mnoznik rozmiaru warstw "feedforward" (przetwarzanie po aplikacji mechanizmu uwagi)

# Parametry procesu nauczania
rozmiar_porcji=120          # Liczba zapytań modelu w pojedynczym kroku uczenia sieci (GPT-3: 3200000)
stopa_optymalizacji=3e-4    # Amplituda korekt wag po każdym wyliczeniu błędu (GPT-3: 0.6e-4)
iteracje=200                 # Liczba iteracji uczenia

# Parametry użytkowania modelu
dlugosc_odpowiedzi=500      # ile tokenów ma znaleźć się w generowanej odpowiedzi