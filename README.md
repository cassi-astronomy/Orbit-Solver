# Orbit Solver

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qt / PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/)
[![Version 1.0](https://img.shields.io/badge/version-1.0-1677ff)](https://github.com/cassi-astronomy/Orbit-Solver)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Desktopová aplikace pro načítání astrometrie Minor Planet Center, určování drah
planetek a komet, tvorbu efemerid a vizualizaci jejich pohybu. Program má české
uživatelské rozhraní a běží ve Windows i Linuxu.

> **English summary:** Orbit Solver is a cross-platform Python/PySide6 desktop
> application for fitting asteroid and comet orbits from MPC astrometry,
> generating ephemerides, inspecting residuals and visualising trajectories.
> The user interface is currently in Czech.

## Hlavní možnosti

### Astrometrie a určení dráhy

- načtení výchozí kartézské dráhy z Minor Planet Center;
- samostatné stažení pozorování z MPC nebo NEOCP/PCCP;
- import pozorování ze souboru a vložení ze schránky ve formátu MPC 80-column;
- podpora družicových pozorování a topocentrické paralaxy podle kódu stanice MPC;
- vlastní šestiparametrový fit kartézského stavu s planetárními perturbacemi;
- rychlý gravitační fit velkých souborů z časově i početně reprezentativního
  vzorku; u krátkého objevitelského oblouku se finální větev ověří na všech
  měřeních a dlouhé negravitační řešení používá větší reprezentativní vzorek;
- výpočet z celého oblouku nebo pouze z posledních 2, 5 či 10 let;
- ruční zařazení a vyřazení jednotlivých pozorování;
- barevná rezidua O−C, automatické vyřazení reziduí nad 2″ a funkce `Worst obs`;
- individuální nejistoty ADES a konzervativní epochové váhy pro starší záznamy
  bez sigmy; moderní chybná CCD měření nadále zachytává robustní fit podle O−C;
- bezpečný víceapariční režim periodických komet: bez A1–A3 se fituje poslední
  pozorovaný návrat a ostatní se označí jako čistě gravitační integrace;
  při zapnutém negravitačním modelu slouží poslední návrat jako počáteční
  korekce pro společný fit celého oblouku;
- volitelné fixování oskulačních elementů;
- volitelný fit negravitačních parametrů komet A1, A2, A3 a časového posunu
  aktivity DT; kladné DT posouvá maximum aktivity za perihel;
- nejistoty elementů a základní diagnostika fitu.

### Elementy a efemerida

- přehled orbitálních elementů včetně nejistot;
- u periodické komety s dlouhým obloukem automatická tabulka integrovaných
  oskulačních elementů pozorovaných návratů a prvního následujícího návratu;
- převod elementů na uživatelem zvolenou epochu;
- přepočet oskulačních elementů vůči Slunci nebo zvolené planetě;
- kopírovatelný textový výstup a MPC 1-line elements;
- geocentrická nebo topocentrická efemerida pro zvolenou MPC stanici;
- RA/Dec v desetinném i sexagesimálním formátu;
- azimut, výška objektu, výška Slunce, vzdálenosti a elongace;
- formální poziční nejistota 1σ v obloukových sekundách;
- kopírování obdélníkového výběru z tabulky.

### Vizualizace

- interaktivní heliocentrický pohled na Sluneční soustavu s otáčením a plynulým
  logaritmickým přiblížením;
- volitelná průsvitná projekční plocha dráhy k rovině ekliptiky pro názorné
  zobrazení sklonu, výšky nad/pod ekliptikou a uzlů dráhy;
- rovníková mapa oblohy, středová azimutální projekce a obzorový graf viditelnosti;
- hvězdy do 6. magnitudy, obrazce a latinské zkratky 88 souhvězdí, rovník a ekliptika;
- volné posouvání mapy, sledování objektu a samostatné zvětšené okno;
- světelná křivka pro planetky i komety s nastavitelným fotometrickým modelem
  a souběžnými křivkami vzdálenosti od Země a od Slunce; v grafu je vyznačeno
  také heliocentrické přísluní.

### Dlouhodobý vývoj dráhy

- integrace podle počtu minulých/budoucích návratů nebo podle samostatného počtu
  let v obou směrech;
- tabulka oskulačních elementů v jednotlivých perihelových průchodech, označených
  přímo rokem návratu;
- hledání přiblížení k planetám, vzdálenost v AU a Hillových poloměrech,
  relativní rychlost a orientační změna `1/a`;
- automatická detekce planetocentrického zachycení podle záporné relativní
  orbitální energie uvnitř Hillovy sféry;
- přepínatelný graf průběžného vývoje `q`, `a`, `e`, periody, `1/a`, sklonu,
  uzlu nebo argumentu perihelu na lineární časové ose; změny při setkáních se
  projeví v okamžiku setkání a setkání jsou v grafu vyznačena;
- chronologický export jako čitelný tabulkový TXT nebo MPC 1-line elements;
- zobrazení numericky integrované minulé a budoucí trajektorie výrazně odlišnými
  barvami ve Sluneční soustavě; animovaná poloha sleduje i změny dráhy po
  planetárním setkání;
- volitelný lokální planetocentrický pohled pro zachycené objekty; lunární
  perturbace jsou zahrnuty (z DE440, nebo přibližným modelem bez něj).
- fragmenty komety Shoemaker–Levy 9 lze načíst jejich úplným označením, například
  `D/1993 F2-A`; pokud MPC neposkytne kartézský stav, program automaticky použije
  historický heliocentrický ekliptikální vektor příslušného fragmentu z JPL Horizons.
  Pro obecné označení `D/1993 F2` slouží jako reprezentativní počáteční odhad
  stav fragmentu G; rozpadlá kometa už nemá jedinou společnou dráhu všech fragmentů.

## Požadavky

- Python 3.10 nebo novější;
- připojení k internetu pro dotazy do MPC;
- balíčky uvedené v [`requirements.txt`](requirements.txt): PySide6,
  Matplotlib, NumPy, SciPy, Requests a Astropy.

Hvězdný katalog, obrazce souhvězdí a katalog observatoří jsou součástí
repozitáře. Pro samotné zobrazení již načtených dat tedy nejsou stahovány.

## Instalace

Nejprve naklonujte repozitář:

```bash
git clone https://github.com/cassi-astronomy/Orbit-Solver.git
cd Orbit-Solver
```

### Windows

V PowerShellu:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

Po instalaci lze program spouštět také dvojklikem na `Orbit.cmd`, pokud příkaz
`python` ukazuje na prostředí s nainstalovanými závislostmi.

Pokud PowerShell odmítne aktivovat virtuální prostředí, lze pro aktuální okno
povolit lokální skripty:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Linux

Na distribucích Debian/Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv libxcb-cursor0
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 run.py
```

Na jiných distribucích použijte odpovídající balíčky pro Python, `venv` a Qt 6.
PySide6 si většinu knihoven Qt přinese ve vlastním wheelu. Pokud se pod
Waylandem objeví problém s grafickým backendem, lze zkusit X11/XWayland:

```bash
QT_QPA_PLATFORM=xcb python3 run.py
```

## Základní pracovní postup

1. Do pole **Objekt** zadejte označení nebo název, například `9841`,
   `C/2024 Y1` nebo `Lovejoy`.
2. Tlačítkem **Načíst objekt** načtěte výchozí dráhu MPC. Pokud jednomu názvu
   odpovídá více těles, program nabídne jejich výběr v pořadí objevu.
3. Astrometrii načtěte samostatně z MPC, textového souboru nebo ze schránky.
   U objektu na NEOCP/PCCP program stáhne pozorování a sestaví vlastní počáteční
   odhad, protože katalogová dráha ještě nemusí existovat.
4. Podle potřeby omezte časový rozsah, vyřaďte chybná měření a zvolte
   **Spočítat vlastní dráhu**.
   Automatický fit používá nejistoty z ADES, kontroluje odlehlá měření po
   celých pozorovacích sériích a vyřazené řádky viditelně odškrtne v tabulce.
   U krátkého oblouku s alespoň šesti měřeními nejprve rychle prosije několik
   odlišných vzdálenostních větví (i když už existuje předběžný odhad), nejlepší
   dopočítá ze všech pozorování a až potom smí
   vyřazovat skutečně velká rezidua. Porovná také volnou a téměř parabolickou
   větev, aby pár chybných pozdních měření nestáhl řešení do jiné lokální dráhy.
   Pokud je oblouk příliš krátký nebo běžný počáteční odhad selže, použijte
   **Předběžná dráha z krátkého oblouku**. Doporučené systematické hledání
   prohledává vzdálenost a radiální rychlost a od šesti měření provede také
   robustní diferenciální korekci; alternativní Väisälova metoda vytvoří
   hypotézu pro ručně zadanou perihelovou vzdálenost `q`.
5. Výsledek prohlížejte v záložkách **Elementy**, **Efemerida**,
   **Vývoj dráhy**, **Sluneční soustava**, **Pohled ze Země** a
   **Světelná křivka**.

Výpočty fitu, pozičních nejistot a dlouhodobé integrace běží na pozadí, aby
uživatelské rozhraní zůstalo ovladatelné.

## Přesnost a omezení

Orbit Solver je pracovní a vizualizační nástroj. Výsledky je u důležitých
aplikací nutné porovnat s MPC, JPL Horizons nebo jiným nezávislým softwarem.

- N-body model zahrnuje perturbace osmi planet. Bez volitelného kernelu používá
  jejich analytické střední elementy; v **Vlastním nastavení** lze jedním
  kliknutím stáhnout a aktivovat JPL DE440 (přibližně 114 MB, 1549–2650).
  Výpočty mimo pokrytí aktivního kernelu program odmítne místo tichého přechodu
  na méně přesný model.
- Samostatný Měsíc je zahrnut; Pluto, relativistické členy a perturbace hmotných
  planetek zatím v dynamickém modelu zahrnuty nejsou.
- Konstantní A1/A2/A3/DT při integraci komety přes mnoho návratů jsou modelový
  předpoklad; skutečná negravitační aktivita komety se může mezi návraty měnit.
  DT je vhodné uvolnit až pro dlouhý oblouk, protože na jednom krátkém zjevení
  bývá silně korelované hlavně s A2.
- Velmi těsné průlety u povrchu Jupiteru zatím nezahrnují jeho zploštění ani
  Galileovy měsíce. Rekonstrukce rozpadu Shoemaker–Levy 9 proto zůstává
  orientační, i když planetocentrický rámec a bodová gravitace Jupiteru fungují.
- Poziční nejistoty jsou formálně propagovány z kovariance fitovaného stavu;
  nezahrnují všechny možné systematické chyby měření ani modelu.
- Jediná předběžná dráha z velmi krátkého oblouku není jednoznačné řešení.
  Slouží jako počáteční odhad pro diferenciální korekci; efemerida se v tomto
  režimu přednastaví po 10 minutách na následujících 48 hodin a její skutečná
  nejistota může rychle růst.
- Fotometrická předpověď je modelová. U komet se skutečná aktivita a jasnost
  mohou od vypočtené křivky výrazně lišit.
- Velmi dlouhé integrace a velké soubory pozorování mohou trvat desítky sekund
  nebo déle podle procesoru a zvoleného modelu.

## Testy

Po instalaci závislostí spusťte:

```bash
python -m unittest discover -s tests -v
```

V Linuxu lze místo `python` použít `python3`.

## Struktura projektu

```text
Orbit-Solver/
├── orbit_app/
│   ├── main_window.py       # Qt rozhraní a vizualizační panely
│   ├── mpc.py               # MPC/NEOCP astrometrie a MPC80 parser
│   ├── mpc_orbit.py         # načtení MPC drah a převod elementů
│   ├── orbit_fit.py         # diferenciální korekce a rezidua
│   ├── orbit_engine.py      # numerická N-body propagace
│   ├── orbit_evolution.py   # návraty a planetární přiblížení
│   ├── comet_dynamics.py    # original/future barycentrická dráha komet
│   ├── observatories.py     # stanice, paralaxa a obzorové souřadnice
│   ├── preview_orbit.py     # geometrie efemerid a náhledů
│   ├── photometry.py        # modely jasnosti
│   └── data/                # offline katalogy hvězd, souhvězdí a stanic
├── tests/
├── run.py
├── Orbit.cmd
└── requirements.txt
```

## Data a služby třetích stran

- Orbitální a astrometrická data se načítají z rozhraní
  [Minor Planet Center](https://www.minorplanetcenter.net/).
- Volitelná planetární efemerida DE440 se stahuje přímo z oficiálního
  [archivu NASA/NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/).
  Program před aktivací ověří její zveřejněný kontrolní součet.
- Hvězdný katalog a obrazce souhvězdí pocházejí z projektu
  [d3-celestial](https://github.com/ofrohn/d3-celestial) a jsou distribuovány
  pod licencí BSD-3-Clause. Její text je uložen v
  [`orbit_app/data/D3_CELESTIAL_LICENSE.txt`](orbit_app/data/D3_CELESTIAL_LICENSE.txt).

## Autor a poděkování

**Autor:** [Martin Mašek](https://github.com/cassi-astronomy)

Program byl vyvíjen s pomocí OpenAI Codex. Astronomická data poskytuje Minor
Planet Center; katalog hvězd a obrazce souhvězdí pocházejí z d3-celestial.

## Vývoj a plánované kroky

Následovat může samostatný Měsíc, relativistické členy, hmotné planetky,
ukládání projektů a distribuční balíčky pro Windows a Linux.

Příspěvky a hlášení reprodukovatelných chyb jsou vítány prostřednictvím
[GitHub Issues](https://github.com/cassi-astronomy/Orbit-Solver/issues). K hlášení
je užitečné přiložit označení objektu, zdroj pozorování, zvolené nastavení,
očekávaný výsledek a soubor `orbit-crash.log`, pokud vznikl.

## Licence

Zdrojový kód je dostupný pod svobodnou licencí [MIT](LICENSE). Copyright
© 2026 Martin Mašek.

Licence přibalených dat d3-celestial je uvedena samostatně v adresáři
`orbit_app/data`.
