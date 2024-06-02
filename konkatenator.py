import sys
import os
import re

if "katalog" in sys.argv:
    katalog=sys.argv[sys.argv.index("katalog")+1]
else:
    print("Brak wskazania katalogu z plikami źródłowymi (parametr: katalog)")
    exit(0)

if "plik" in sys.argv:
    plik=sys.argv[sys.argv.index("plik")+1]
else:
    print("Brak wskazania pliku docelowego (parametr: plik)")
    exit(0)

regex=re.compile(r"^[a-zA-Z0-9 .,!?\n()-:;'\"#|$’—“\[\]%@&<>=]+$")
accepted = [8221,8216,8230,8211,163,95,8213,8364,145,183,94,173]
def islatin(plik):
  for znak in plik:
    if (not regex.match(znak)) and (ord(znak) not in accepted):
      return False
  return True

output_file = open(plik, "w")

wlaczone=0
pominiete=0
for file_path in os.listdir(katalog):
  if file_path.endswith(".txt"):
    with open(os.path.join(katalog, file_path), "r") as input_file:
      print("Procesuję "+file_path)
      zawartosc=input_file.read()
      if (islatin(zawartosc)):
        output_file.write("\nArticle starts\n\n")
        output_file.write(zawartosc)
        output_file.write("\n\nArticle ends\n")
        wlaczone+=1
      else:
        pominiete+=1
print("Plików włączonych: "+str(wlaczone)+" pominiętych: "+str(pominiete))

output_file.close()