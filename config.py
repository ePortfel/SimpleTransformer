# Model bliski maksymalnych możliwości karty RTX4080 16GB (GPT-3 ma 175mld parametrów)

# Parametry modelu
tokens=400#1800                 # liczba tokenów w słowniku 
rozmiar_bloku=160#400           # Rozmiar fragmentu tekstu (w tokenach) do analizy kolejnego tokena (GPT-3: 2048)
kanaly_uwagi=4#9              # Liczba równoległych kanałów aplikacji mechanizmu uwagi (GPT-3: 96) 
warstwy=7                   # Liczba warstw przetwarzania (GPT-3: 96)
atrybuty=kanaly_uwagi*192   # Liczba cech kodujących token tekstu (GPT-3: 12288)

# Parametry procesu nauczania
rozmiar_porcji=34           # Liczba zapytań modelu w pojedynczym kroku uczenia sieci (GPT-3: 3200000)
stopa_optymalizacji=3e-4    # Amplituda korekt wag po każdym wyliczeniu błędu (GPT-3: 0.6e-4)
iteracje=4000                # Liczba iteracji uczenia

# Parametry użytkowania modelu
dlugosc_odpowiedzi=500      # ile tokenów ma znaleźć się w generowanej odpowiedzi