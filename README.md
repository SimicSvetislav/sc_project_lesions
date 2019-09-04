# sc_project_lesions
Projekat iz predmeta Soft kompjuting čija je tema klasifikacija mladeža na osnovu njihovih medicinskih slika.

Grana sa postavkom za generisanje exe fajla.
Potrebno je imati instaliran paket pyinstaller. To je moguće uraditi pomoću pip-a: `pip install pyinstaller`  
Zatim je potrebno generisati izvršni fajl:  
`pyinstaller --hidden-import pywt._extensions._cwt main.py`

Izvršni fajl se nalazi na putanji sc_project_lesions/dist/main/main.exe

Fajl u kome je naveden naziv slike iz testnog skupa sa odgovarajućom predikcijom se nalazi na putanji sc_project_lesions/dist/main/predictions.txt
