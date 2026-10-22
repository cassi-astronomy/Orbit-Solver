# Orbit Solver

**Česky** | [English](README.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qt / PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/)
[![Version 1.0](https://img.shields.io/badge/version-1.0-1677ff)](https://github.com/cassi-astronomy/Orbit-Solver)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Desktopová aplikace pro načítání astrometrie Minor Planet Center, určování drah
planetek a komet, tvorbu efemerid a vizualizaci jejich pohybu. Program má anglické a české
uživatelské rozhraní a běží ve Windows i Linuxu.

Výchozím jazykem je angličtina. Češtinu zvolíte v levém panelu přepínačem
**Language / Jazyk**; program se restartuje v novém jazyce a volbu si
zapamatuje i pro další spuštění.

## Hlavní možnosti

### Astrometrie a určení dráhy

- načtení výchozí kartézské dráhy z Minor Planet Center;
- samostatné stažení pozorování z MPC nebo NEOCP/PCCP;
- místní cache astrometrie z MPC: jednou stažená pozorování objektu (např.
  1P) se uloží do `orbit-data/mpc-cache` a příště se načtou okamžitě bez
  dotazu do MPC; tlačítko **Aktualizovat astrometrii z MPC** stáhne aktuální
  data včetně nových pozorování a cache přepíše (objekty z NEOCP/PCCP se
  kvůli rychlým změnám neukládají);
- import pozorování ze souboru a vložení ze schránky ve formátu MPC 80-column;
- podpora družicových pozorování a topocentrické paralaxy podle kódu stanice MPC;
- vlastní šestiparametrový fit kartézského stavu s planetárními perturbacemi;
- rychlý fit velkých souborů z časově i početně reprezentativního vzorku;
  dlouhé oblouky, negravitační řešení a husté objevitelské oblouky se
  závěrečně korigují na všech povolených měřeních;
- výpočet z celého oblouku nebo pouze z posledních 2, 5 či 10 let;
- ruční zařazení a vyřazení jednotlivých pozorování;
- barevná rezidua O−C, automatické vyřazení reziduí nad 2″ a funkce `Worst obs`;
- import historických 80sloupcových záznamů s nižší přesností (RA na celé
  sekundy či minuty, Dec na desetiny minuty); jejich nejistota nikdy není
  menší než polovina posledního publikovaného řádu;
- individuální nejistoty ADES (s dolní mezí 0,3″ kvůli katalogovým a
  časovým systematikám); u měření bez nich výchozí nejistota podle
  techniky (mikrometr 3″, fotografická deska 1,5–2,5″, CCD 1–2″, jinak podle
  epochy); série jedné stanice z jedné místní noci se převáží pravidlem
  `sqrt(N/4)` podle Vereše et al. (2017), aby stovky snímků nepředstíraly
  stovky nezávislých absolutních poloh;
- duplicitní redukce téhož měření (MPC je často uvádí vůči více hvězdným
  katalogům, např. PPM a UCAC-4) se rozpoznají a fituje se jen ta s nejlepším
  katalogem; záznamy, které MPC označuje jako nahrazené (`X`, ADES
  `deprecated`), se nefitují;
- časy před rokem 1960 jsou světový čas UT a na TT se převádějí podle ΔT
  z rekonstrukce rotace Země (Espenak & Meeus), ne pevným posunem 32,184 s;
- navazování vzdálených návratů pro jakékoli těleso (periodické komety,
  víceopoziční planetky): pozorování se rozdělí na aparice podle mezer delších
  než polovina oběžné doby (nejméně 300 dní); pokud výchozí dráha všechny
  aparice předpovídá, opraví se společně; jinak se samostatně napasuje nejlépe
  pozorovaný návrat (z katalogové dráhy, případně z vlastních úhlů postupným
  rozšiřováním oblouku) a další návraty se připojují po jednom, časově
  nejbližší první; když předpověď pro diferenciální korekci nestačí, hledá se
  správná větev podél linie variací (LOV) – nejslabšího směru kovariance, tedy
  rodiny drah, které stále sedí na již navázanou astrometrii; každý návrat je
  přijat, jen pokud sedí na vlastní astrometrii (normalizované RMS ≤ 3σ po
  omezeném a viditelném vyřazení hrubých chyb); dříve navázané návraty se
  přitom smějí posunout nejvýše do 6σ a takové napětí modelu (typicky
  proměnlivá aktivita komety) program výslovně vypíše; návraty, které již
  navázaná dráha dobře předpovídá, se připojí najednou v jedné korekci;
  požadované A1/A2 se fitují, jakmile oblouk pokrývá celý oběh (dvě perihelia),
  ostatní požadované parametry v závěrečném společném řešení; takto se např. 206P/Barnard–Boattini (1892
  a 2008, 20 oběhů a tři přiblížení k Jupiteru) naváže i bez katalogové dráhy;
- integrace oblouků delších než dva roky používá přesnou toleranci
  (`rtol = 1e-12`); přes staletí a těsná setkání s Jupiterem by volnější
  tolerance vnesla chybu stovek obloukových vteřin;
- volitelné fixování oskulačních elementů;
- předem volitelná výstupní epocha; fit zůstává kvůli stabilitě uprostřed
  pozorovacího oblouku a výsledné elementy se následně převedou N-body integrací;
- volitelný fit negravitačních parametrů A1, A2 a A3 pro libovolné těleso:
  Marsdenův vodní model s volitelným časovým posunem aktivity DT pro komety,
  nebo mocninný model `g(r) = r^-m` (výchozí `m = 2`) pro tlak záření a
  Yarkovského A2 u planetek; označení tělesa funkci neomezuje;
- nejistoty elementů a základní diagnostika fitu; při singulární nebo extrémně
  špatně podmíněné normální matici se nejistota raději neuvede, než aby
  pseudoinverze vytvořila falešně přesné číslo.

### Elementy a efemerida

- přehled orbitálních elementů včetně nejistot;
- u navázaných návratů (a u komety s dlouhým obloukem) spočte fit jednu
  integraci od prvního pozorovaného návratu po první návrat po dnešku; Vývoj
  dráhy ji ihned zobrazí celou – oskulační elementy v každém návratu,
  přiblížení k planetám, průběh elementů v grafu i trajektorii ve Sluneční
  soustavě – bez nového spouštění integrace; tlačítko Spustit integraci slouží
  jen pro jiný rozsah;
- převod elementů na uživatelem zvolenou epochu;
- přepočet oskulačních elementů vůči Slunci nebo zvolené planetě;
- kopírovatelný textový výstup a MPC 1-line elements;
- geocentrická nebo topocentrická efemerida pro zvolenou MPC stanici;
- numerická N-body propagace z kartézského stavu i u čistě gravitační dráhy,
  časový argument TDB a iterovaná korekce na dobu letu světla;
- astrometrické RA/Dec v soustavě ICRS/J2000 v desetinném i sexagesimálním
  formátu; převod na obzorové souřadnice zahrnuje precesi, nutaci a aberaci;
- azimut, výška objektu, výška Slunce, vzdálenosti a elongace;
- formální poziční nejistota 1σ v obloukových sekundách;
- kopírování obdélníkového výběru z tabulky.

### Vizualizace

- interaktivní heliocentrický pohled na Sluneční soustavu s otáčením a plynulým
  logaritmickým přiblížením;
- oskulační křivka tělesa se odvozuje z N-body stavu aktuálního snímku, takže
  animovaný bod neleží mimo starou křivku z epochy původního fitu;
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
  barvami ve Sluneční soustavě; právě probíhající návrat je během animace
  zvýrazněn silnější světlou čarou, zatímco starší a budoucí revoluce zůstávají
  tenké a tlumené; významné planetární setkání je označené jako dynamická
  „výhybka“ a v okamžiku průletu se zvýraznění přepne z příchozí na odchozí
  větev; animovaná poloha sleduje i skutečnou změnu dráhy po setkání;
- volitelný lokální planetocentrický pohled pro zachycené objekty; lunární
  perturbace jsou zahrnuty (z DE440, nebo přibližným modelem bez něj).
- fragmenty komety Shoemaker–Levy 9 lze načíst jejich úplným označením, například
  `D/1993 F2-A`; pokud MPC neposkytne kartézský stav, program automaticky použije
  historický heliocentrický ekliptikální vektor příslušného fragmentu z JPL Horizons.
  Pro obecné označení `D/1993 F2` slouží jako reprezentativní počáteční odhad
  stav fragmentu G; rozpadlá kometa už nemá jedinou společnou dráhu všech fragmentů.

## Požadavky

- Python 3.10 nebo novější;
- připojení k internetu pro dotazy do MPC (astrometrie uložená v cache funguje
  i offline);
- balíčky uvedené v [`requirements.txt`](requirements.txt): PySide6,
  Matplotlib, NumPy, SciPy, Requests, Astropy a jplephem.

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
3. Astrometrii načtěte samostatně tlačítkem **Načíst astrometrii z MPC**,
   z textového souboru nebo ze schránky. Pokud už byla astrometrie objektu
   dříve stažena, načte se z místní cache (stavový řádek uvede datum stažení);
   nová pozorování získáte tlačítkem **Aktualizovat astrometrii z MPC**.
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

- N-body model zahrnuje perturbace osmi planet a systému Pluto. Bez volitelného kernelu používá
  jejich analytické střední elementy; v **Vlastním nastavení** lze jedním
  kliknutím stáhnout a aktivovat JPL DE440 (přibližně 114 MB, 1549–2650).
  Výpočty mimo pokrytí aktivního kernelu program odmítne místo tichého přechodu
  na méně přesný model.
- Samostatný Měsíc i první postnewtonovská (Schwarzschildova) korekce Slunce
  jsou zahrnuty. Úplný EIH/PPN model a perturbace 16 hmotných planetek z
  `sb441-n16.bsp` zatím zahrnuty nejsou; tento přibližně 650MB asteroidový
  kernel vyžaduje čtečku SPK typu používaného SB441, kterou současná knihovna
  `jplephem` neposkytuje.
- Navazování návratů předpokládá jeden společný silový model. Návrat, který
  společné řešení nepotvrdí, program výslovně označí a zobrazí jen jako
  předpověď; to samo o sobě identitu návratů nevyvrací.
- Systematické chyby hvězdných katalogů (debiasing podle Farnocchii et al.)
  zatím korigovány nejsou; u historických měření se projevují jako rozdíly
  mezi stanicemi v řádu obloukových vteřin.
- Konstantní A1/A2/A3/DT při integraci komety přes mnoho návratů jsou modelový
  předpoklad; skutečná negravitační aktivita komety se může mezi návraty měnit.
  DT je vhodné uvolnit až pro dlouhý oblouk, protože na jednom krátkém zjevení
  bývá silně korelované hlavně s A2.
- Velmi těsné průlety u povrchu Jupiteru zatím nezahrnují jeho zploštění ani
  Galileovy měsíce. Rekonstrukce rozpadu Shoemaker–Levy 9 proto zůstává
  orientační, i když planetocentrický rámec a bodová gravitace Jupiteru fungují.
- Poziční nejistoty jsou formálně propagovány z kovariance fitovaného stavu;
  nezahrnují všechny možné systematické chyby měření ani modelu.
- RA/Dec v tabulce efemeridy jsou záměrně astrometrické ICRS/J2000 souřadnice
  opravené o dobu letu světla, tedy protějšek astrometrického výsledku
  `observe()` ve Skyfieldu. Nejde o zdánlivé RA/Dec k aktuálnímu ekvinokciu;
  pro přímé navádění dalekohledu je nutný převod na zdánlivé souřadnice nebo
  na zobrazené topocentrické hodnoty azimutu a výšky. Atmosférická refrakce se
  bez místního tlaku a teploty neodhaduje.
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

Pro vývoj lze projekt nainstalovat v editovatelném režimu včetně statické
kontroly Ruff:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m unittest discover -s tests -v
```

Stejné kontroly automaticky spouští GitHub Actions pro každý push a pull
request. Testy nepoužívají živé dotazy do MPC/JPL ani nestahují DE440; síťová
rozhraní a nákladné numerické větve jsou podle potřeby nahrazeny testovacími
daty nebo mocky. Čisté formátovací a projekční funkce jsou odděleny v modulu
`orbit_app/presentation.py`, který nemá závislost na Qt ani Matplotlibu.
Na systému bez PySide6 lze spustit alespoň oddělenou kontrolu jádra:

```bash
python -m unittest tests.test_headless -v
```

## Struktura projektu

```text
Orbit-Solver/
├── orbit_app/
│   ├── main_window.py       # Qt rozhraní a vizualizační panely
│   ├── mpc.py               # MPC/NEOCP astrometrie a MPC80 parser
│   ├── mpc_cache.py         # místní cache stažené astrometrie MPC
│   ├── mpc_orbit.py         # načtení MPC drah a převod elementů
│   ├── astrometry.py        # váhy měření, duplicitní redukce, normální místa
│   ├── orbit_fit.py         # diferenciální korekce a rezidua
│   ├── linkage.py           # navazování návratů hledáním po linii variací
│   ├── orbit_engine.py      # numerická N-body propagace
│   ├── orbit_evolution.py   # návraty a planetární přiblížení
│   ├── comet_dynamics.py    # original/future barycentrická dráha komet
│   ├── observatories.py     # stanice, paralaxa a obzorové souřadnice
│   ├── presentation.py      # čisté formátování a projekce bez závislosti na Qt
│   ├── preview_orbit.py     # geometrie efemerid a náhledů
│   ├── photometry.py        # modely jasnosti
│   ├── i18n.py              # volba jazyka rozhraní a funkce tr()
│   ├── translations_en.py   # anglický katalog textů rozhraní
│   └── data/                # offline katalogy hvězd, souhvězdí a stanic
├── .github/workflows/       # automatické kontroly Ruff a unittest
├── tests/
├── pyproject.toml           # metadata projektu a konfigurace vývojových nástrojů
├── run.py
├── Orbit.cmd
└── requirements.txt
```

## Data a služby třetích stran

- Orbitální a astrometrická data se načítají z rozhraní
  [Minor Planet Center](https://www.minorplanetcenter.net/). Stažená
  astrometrie se ukládá do `orbit-data/mpc-cache` (gzip JSON, jeden soubor na
  objekt); cache lze kdykoli smazat, program si data při potřebě stáhne znovu.
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

Následovat mohou úplný EIH/PPN model, hmotné planetky, ukládání projektů
a distribuční balíčky pro Windows a Linux.

Příspěvky a hlášení reprodukovatelných chyb jsou vítány prostřednictvím
[GitHub Issues](https://github.com/cassi-astronomy/Orbit-Solver/issues). K hlášení
je užitečné přiložit označení objektu, zdroj pozorování, zvolené nastavení,
očekávaný výsledek a soubor `orbit-crash.log`, pokud vznikl.

## Licence

Zdrojový kód je dostupný pod svobodnou licencí [MIT](LICENSE). Copyright
© 2026 Martin Mašek.

Licence přibalených dat d3-celestial je uvedena samostatně v adresáři
`orbit_app/data`.
