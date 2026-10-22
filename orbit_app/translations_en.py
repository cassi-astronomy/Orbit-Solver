"""English texts for the Czech source templates used with :func:`orbit_app.i18n.tr`.

Keys are the exact Czech templates from the code (``str.format`` syntax with
positional fields); values must use the same fields.  Keep entries sorted.
"""

CATALOG: dict[str, str] = {
    "\n\nPodrobný protokol je v záložce Elementy.":
        "\n\nThe detailed log is in the Elements tab.",
    "\nAutomaticky vyřazeno: {0}":
        "\nRejected automatically: {0}",
    "    Az {0:.1f}°    výška {1:+.1f}°    Slunce {2:+.1f}°    stanice {3}":
        "    Az {0:.1f}°    alt {1:+.1f}°    Sun {2:+.1f}°    station {3}",
    "    a = {0:.1f} AU; přibližná perioda {1:.0f} let":
        "    a = {0:.1f} AU; approximate period {1:.0f} years",
    "    od {0} {1:.6f} AU":
        "    from {0} {1:.6f} AU",
    "  Detekováno planetocentrické zachycení: ":
        "  Planetocentric capture detected:",
    "  budoucí":
        "  future",
    " a následně original/future 1/a":
        " and then original/future 1/a",
    " z MPC":
        " from the MPC",
    " z NEOCP/PCCP":
        " from NEOCP/PCCP",
    " • {0} pozorování":
        " • {0} observations",
    "# Elementy jsou oskulační v okamžiku příslušného perihelu.":
        "# Elements are osculating at the moment of the respective perihelion.",
    "# Návrat {0:+d} — {1} UTC":
        "# Return {0:+d} — {1} UTC",
    "# Návrat {0:+d}: kalendářní datum je mimo podporovaný rozsah.":
        "# Return {0:+d}: the calendar date is outside the supported range.",
    "# Orbit Solver 1.0 — vývoj dráhy: {0}":
        "# Orbit Solver 1.0 — orbit evolution: {0}",
    "# Rozsah UTC: {0} až {1}":
        "# UTC range: {0} to {1}",
    "# SETKÁNÍ {0} UTC — {1}; d={2:.6f} AU = {3:.3f} Hillova poloměru; v_rel={4:.2f} km/s; Delta(1/a)={5}":
        "# ENCOUNTER {0} UTC — {1}; d={2:.6f} AU = {3:.3f} Hill radii; v_rel={4:.2f} km/s; Delta(1/a)={5}",
    "# STAV {0} — {1} UTC; q={2:.9f} AU; e={3:.10f}; a={4:.9f} AU; 1/a={5:+.10e} AU^-1; i={6:.7f}; Omega={7:.7f}; omega={8:.7f}":
        "# STATE {0} — {1} UTC; q={2:.9f} AU; e={3:.10f}; a={4:.9f} AU; 1/a={5:+.10e} AU^-1; i={6:.7f}; Omega={7:.7f}; omega={8:.7f}",
    "# ZACHYCENÍ {0}: {1} až {2} UTC; minimum={3:.6f} AU ({4:.3f} Hillova poloměru)":
        "# CAPTURE {0}: {1} to {2} UTC; minimum={3:.6f} AU ({4:.3f} Hill radii)",
    "# Řádky začínající # jsou komentáře a planetární setkání.":
        "# Lines starting with # are comments and planetary encounters.",
    "+X  ♈ jarní bod":
        "+X  ♈ vernal equinox",
    "+Z  sever ekliptiky":
        "+Z  north ecliptic pole",
    ", vážené RMS {0:.2f}σ":
        ", weighted RMS {0:.2f}σ",
    "1 den":
        "1 day",
    "1 den/s":
        "1 day/s",
    "1 hodina":
        "1 hour",
    "10 dní/s":
        "10 days/s",
    "10 minut":
        "10 minutes",
    "100 dní/s":
        "100 days/s",
    "12 hodin celkem":
        "12 hours in total",
    "14 dní celkem":
        "14 days in total",
    "15 dní/s":
        "15 days/s",
    "2 dny celkem":
        "2 days in total",
    "2 dny/s":
        "2 days/s",
    "20 dní/s":
        "20 days/s",
    "24 hodin celkem":
        "24 hours in total",
    "25 dní/s":
        "25 days/s",
    "250 dní/s":
        "250 days/s",
    "3 dny/s":
        "3 days/s",
    "30 minut":
        "30 minutes",
    "5 dní/s":
        "5 days/s",
    "50 dní/s":
        "50 days/s",
    "500 dní/s":
        "500 days/s",
    "6 dní celkem":
        "6 days in total",
    "; original/future 1/a dopočítáno":
        "; original/future 1/a computed",
    "; original/future výpočet se nezdařil":
        "; the original/future calculation failed",
    "A1 radiální [10⁻⁸ AU/den²]":
        "A1 radial [10⁻⁸ AU/day²]",
    "A1–A3 a volitelný časový posun DT se fitují jako volné parametry. DT má smysl především u více návratů; kladná hodnota znamená maximum aktivity po perihelu.":
        "A1–A3 and the optional activity time shift DT are fitted as free parameters. DT is meaningful mainly for several returns; a positive value means peak activity after perihelion.",
    "A1–A3 lze fitovat bez ohledu na označení tělesa. Pro planetky je obvyklá příčná A2 (Yarkovsky); radiální A1 může popsat tlak záření. Marsdenův model a DT jsou určeny pro kometární odplyňování. Detekce Yarkovského efektu vyžaduje dlouhý přesný oblouk; samotná nenulová hodnota A2 na krátkém oblouku není průkazná.":
        "A1–A3 can be fitted regardless of the body's designation. For asteroids the transverse A2 (Yarkovsky) is usual; the radial A1 can describe radiation pressure. The Marsden model and DT are intended for cometary outgassing. Detecting the Yarkovsky effect requires a long precise arc; a non-zero A2 on a short arc alone is not conclusive.",
    "A2 příčný [10⁻⁸ AU/den²]":
        "A2 transverse [10⁻⁸ AU/day²]",
    "A3 normálový [10⁻⁸ AU/den²]":
        "A3 normal [10⁻⁸ AU/day²]",
    "Aktivní vlastní nastavení: {0}":
        "Active custom settings: {0}",
    "Alespoň jeden směr musí mít nenulový rozsah.":
        "At least one direction must have a non-zero range.",
    "Astrometrie pro vlastní fit":
        "Astrometry for the own fit",
    "Automatické načtení objektu selhalo.":
        "Automatic loading of the object failed.",
    "Azimut":
        "Azimuth",
    "Azimut kamery":
        "Camera azimuth",
    "Barva podle většího z |O−C RA| a |O−C Dec|: zelená ≤0,5″; světle zelená ≤1″; žlutá ≤2″; červená >2″.":
        "Colour by the larger of |O−C RA| and |O−C Dec|: green ≤0.5″; light green ≤1″; yellow ≤2″; red >2″.",
    "Budoucí návraty:":
        "Future returns:",
    "Celá obloha":
        "Whole sky",
    "Celá obrazovka (F11)":
        "Full screen (F11)",
    "Celý pozorovací oblouk":
        "Whole observed arc",
    "Chronologický TXT výstup byl zkopírován do schránky.  {0}":
        "The chronological TXT output was copied to the clipboard.  {0}",
    "Chronologický přehled (elementy jsou oskulační v uvedené epoše)":
        "Chronological summary (elements are osculating at the given epoch)",
    "Chronologický výstup byl uložen do {0}":
        "The chronological output was saved to {0}",
    "Chybí počáteční kartézský stav dráhy.":
        "The initial Cartesian state of the orbit is missing.",
    "DT":
        "DT",
    "DT zpoždění aktivity [dny]":
        "DT activity delay [days]",
    "Datum":
        "Date",
    "Datum (UTC)":
        "Date (UTC)",
    "Datum a čas uprostřed zobrazeného intervalu v UTC.":
        "Date and time in the middle of the displayed interval in UTC.",
    "Datum a čas uprostřed zobrazeného intervalu v pásmu {0}.":
        "Date and time in the middle of the displayed interval in the {0} zone.",
    "Dec astrom. J2000 (° ′ ″)":
        "Dec astrom. J2000 (° ′ ″)",
    "Diferenciální korekce se vzdálila od astrometrie o celé stupně (RMS {0:.0f}″; {1:.0f}σ). Takové řešení nelze považovat za určenou dráhu.":
        "The differential correction moved whole degrees away from the astrometry (RMS {0:.0f}″; {1:.0f}σ). Such a solution cannot be considered a determined orbit.",
    "Dlouhodobá dynamika komety":
        "Long-term comet dynamics",
    "Dlouhé rozsahy jsou určeny hlavně pro přehled. Přesnost vzdálené extrapolace závisí na nejistotě dráhy a planetárních perturbacích.":
        "Long ranges are mainly for an overview. The accuracy of a distant extrapolation depends on the orbit uncertainty and planetary perturbations.",
    "Do zvoleného časového rozsahu a ručního výběru spadá {0} z {1} načtených měření. Bez katalogové dráhy program určí počáteční stav přímo z astrometrie.":
        "{0} of {1} loaded measurements fall into the chosen time range and manual selection. Without a catalogue orbit the program determines the initial state directly from the astrometry.",
    "Do:":
        "To:",
    "Dokončuji O−C, nejistoty a fotometrii…":
        "Finishing O−C, uncertainties and photometry…",
    "Dočasná planetocentrická zachycení":
        "Temporary planetocentric captures",
    "Dráha":
        "Orbit",
    "Dráha načtena. Grafy a efemerida se přepočítají při otevření záložky.":
        "Orbit loaded. Plots and the ephemeris are recomputed when their tab is opened.",
    "Dráha nedosáhla hranice {0:g} AU.":
        "The orbit did not reach the {0:g} AU boundary.",
    "Dráha nemá kartézský stav.":
        "The orbit has no Cartesian state.",
    "Dvojklikem na návrat jej zobrazíte ve Sluneční soustavě.":
        "Double-click a return to show it in the Solar System.",
    "Efemerida":
        "Ephemeris",
    "Ekliptika":
        "Ecliptic",
    "Elementy":
        "Elements",
    "Elementy převedeny N-body integrací na {0:%Y-%m-%d %H:%M}. Formální nejistoty nebyly do nové epochy propagovány. Planetární efemerida: {1}.":
        "Elements converted by N-body integration to {0:%Y-%m-%d %H:%M}. Formal uncertainties were not propagated to the new epoch. Planetary ephemeris: {1}.",
    "Epoch: kalendářní datum mimo podporovaný rozsah [JDT {0:.8f}]":
        "Epoch: calendar date outside the supported range [JDT {0:.8f}]",
    "Epocha (UTC)":
        "Epoch (UTC)",
    "Epocha fitu":
        "Fit epoch",
    "Excentricita e":
        "Eccentricity e",
    "Exponent negravitačního modelu musí být kladný.":
        "The exponent of the non-gravitational model must be positive.",
    "Fit skončil v numericky nestabilní kombinaci A1/A2/A3/DT. Zkuste vypnout DT nebo některý slabě určený negravitační parametr.":
        "The fit ended in a numerically unstable combination of A1/A2/A3/DT. Try disabling DT or a weakly determined non-gravitational parameter.",
    "Fit zůstane kvůli numerické stabilitě v epoše uprostřed pozorování; výsledné oskulační elementy se po výpočtu převedou na tuto epochu.":
        "For numerical stability the fit stays at an epoch in the middle of the observations; the resulting osculating elements are converted to this epoch afterwards.",
    "Fixace elementů je vynucena během diferenciální korekce. Pro parabolickou hypotézu lze nastavit e = 1. ":
        "Element constraints are enforced during the differential correction. For a parabolic hypothesis set e = 1.",
    "Fixovat objekt ve středu":
        "Keep the object centred",
    "Fáze":
        "Phase",
    "Graf: argument perihelu ω":
        "Plot: argument of perihelion ω",
    "Graf: excentricita e":
        "Plot: eccentricity e",
    "Graf: oběžná doba P":
        "Plot: orbital period P",
    "Graf: perihelová vzdálenost q":
        "Plot: perihelion distance q",
    "Graf: reciproká poloosa 1/a":
        "Plot: reciprocal semi-major axis 1/a",
    "Graf: sklon i":
        "Plot: inclination i",
    "Graf: velká poloosa a":
        "Plot: semi-major axis a",
    "Graf: výstupný uzel Ω":
        "Plot: ascending node Ω",
    "Gravitační parametr středu musí být kladný.":
        "The gravitational parameter of the centre must be positive.",
    "Hammerova projekce celé oblohy":
        "Hammer projection of the whole sky",
    "Heliocentrická ekliptikální soustava J2000 • tažení myší otáčí kameru • kolečko přibližuje":
        "Heliocentric ecliptic frame J2000 • drag with the mouse to rotate the camera • wheel zooms",
    "Hillovy poloměry":
        "Hill radii",
    "Hledám název {0} v MPC…":
        "Looking up the name {0} at the MPC…",
    "Hledám předběžnou dráhu {0}; výpočet běží na pozadí…":
        "Searching for a preliminary orbit {0}; the calculation runs in the background…",
    "Hranice výpočtu: {0:g} AU od Slunce":
        "Calculation boundary: {0:g} AU from the Sun",
    "Hyperbolická univerzální anomálie přetekla.":
        "The hyperbolic universal anomaly overflowed.",
    "H–G fázová korekce":
        "H–G phase correction",
    "Integrace spočtená při vlastním fitu přes pozorované návraty až po první návrat po dnešku; planetární perturbace a použitý silový model jsou stejné jako ve fitu.":
        "Integration computed during the own fit across the observed returns up to the first return after today; planetary perturbations and the force model are the same as in the fit.",
    "Integrace vrátila nekonečný stav.":
        "The integration returned an infinite state.",
    "Integrovaná dráha: vypnuta":
        "Integrated orbit: off",
    "Integrovaná dráha: zapnuta":
        "Integrated orbit: on",
    "Integruji dráhu na epochu {0:%Y-%m-%d %H:%M}… U vzdálené epochy to může trvat desítky sekund.":
        "Integrating the orbit to epoch {0:%Y-%m-%d %H:%M}… For a distant epoch this may take tens of seconds.",
    "Integruji původní a budoucí barycentrickou dráhu do 250 AU…":
        "Integrating the original and future barycentric orbit to 250 AU…",
    "Integruji…":
        "Integrating…",
    "Izolované měření s velkým normalizovaným O−C; po automatickém vyřazení byla dráha znovu spočtena.":
        "Isolated measurement with a large normalized O−C; the orbit was recomputed after its automatic rejection.",
    "JPL DE440 byla ověřena a aktivována. Nové výpočty budou používat planetární efemeridu DE440 v rozsahu 1549–2650.":
        "JPL DE440 was verified and activated. New calculations will use the DE440 planetary ephemeris in the range 1549–2650.",
    "JPL DE440 chybí; používá se méně přesný analytický model planet.":
        "JPL DE440 is missing; the less precise analytic planetary model is used.",
    "JPL DE440 je aktivní ({0:.0f} MB; 1549–2650).":
        "JPL DE440 is active ({0:.0f} MB; 1549–2650).",
    "JPL Horizons (reprezentativní počáteční stav fragmentu G)":
        "JPL Horizons (representative initial state of fragment G)",
    "JPL Horizons dotaz selhal: {0}":
        "JPL Horizons query failed: {0}",
    "JPL Horizons nevrátil epochu fragmentu SL9.":
        "JPL Horizons did not return the epoch of the SL9 fragment.",
    "JPL Horizons stav SL9 nelze načíst: {0}":
        "The JPL Horizons SL9 state cannot be loaded: {0}",
    "JPL Horizons vrátil nečitelná data: {0}":
        "JPL Horizons returned unreadable data: {0}",
    "Je potřeba alespoň jeden konečný výstupní čas.":
        "At least one finite output time is needed.",
    "Je potřeba alespoň jeden výstupní čas.":
        "At least one output time is needed.",
    "K odhadu H/M nejsou k dispozici platná fotometrická data.":
        "No valid photometric data are available to estimate H/M.",
    "Kartézský stav (AU, AU/den):":
        "Cartesian state (AU, AU/day):",
    "Kartézský stav nevede na použitelnou kuželosečku.":
        "The Cartesian state does not lead to a usable conic.",
    "Katalogová dráha není dostupná; hledám pozorování v MPC a NEOCP/PCCP…":
        "No catalogue orbit is available; looking for observations in the MPC and NEOCP/PCCP…",
    "Kernel se nepodařilo otevřít.":
        "The kernel could not be opened.",
    "Kliknutím na pozorování zobrazíte observatoř a geometrii v okamžiku měření.":
        "Click an observation to show the observatory and the geometry at the time of measurement.",
    "Kometa – celková jasnost":
        "Comet – total brightness",
    "Kontrola O−C odhalila numericky nekonzistentní fit; původní dráha zůstala zachována.":
        "The O−C check revealed a numerically inconsistent fit; the original orbit was kept.",
    "Kontrolní integrace O−C skončila v nestabilní kombinaci A1/A2/A3/DT; zkuste vypnout nejslaběji určený parametr.":
        "The O−C check integration ended in an unstable combination of A1/A2/A3/DT; try disabling the most weakly determined parameter.",
    "Kontrolní propagace dráhy skončila numerickou chybou.":
        "The check propagation of the orbit ended with a numerical error.",
    "Kontrolní součet nesouhlasí. Soubor může být neúplný nebo poškozený.":
        "The checksum does not match. The file may be incomplete or damaged.",
    "Kopírovat MPC 1-line elements":
        "Copy MPC 1-line elements",
    "Kopírovat TXT":
        "Copy TXT",
    "Kopírovat přehled":
        "Copy summary",
    "Kopírovat vybraná":
        "Copy selected",
    "Kopírovat vybrané":
        "Copy selected",
    "Kopírovat vše":
        "Copy all",
    "Krok vyhledávání událostí [dny]:\t{0:.6f}":
        "Event search step [days]:\t{0:.6f}",
    "Krok:":
        "Step:",
    "Kurzor:":
        "Cursor:",
    "Kód pozorovací stanice MPC; 557 = Ondřejov. Výška je geometrická, bez refrakce.":
        "MPC observatory code; 557 = Ondřejov. Altitude is geometric, without refraction.",
    "LOV {0:+.2f}σ: medián {1:.1f}σ":
        "LOV {0:+.2f}σ: median {1:.1f}σ",
    "MPC 1-line elements jsou definovány pro heliocentrickou dráhu; pro planetocentrické elementy nejsou exportovány.":
        "MPC 1-line elements are defined for a heliocentric orbit; they are not exported for planetocentric elements.",
    "MPC 1-line elements nelze vytvořit: kalendářní datum epochy nebo perihelia je mimo podporovaný rozsah.":
        "MPC 1-line elements cannot be created: the calendar date of the epoch or perihelion is outside the supported range.",
    "MPC dotaz selhal: {0}":
        "MPC query failed: {0}",
    "MPC dráha je počáteční odhad; vlastní fit používá topocentrické polohy stanic a automatické planetární perturbace.":
        "The MPC orbit is an initial estimate; the own fit uses topocentric station positions and automatic planetary perturbations.",
    "MPC kód observatoře; 500 znamená geocentrickou efemeridu.":
        "MPC observatory code; 500 means a geocentric ephemeris.",
    "MPC neposkytlo počáteční dráhu, proto bude nejprve určena z astrometrie; fixace elementů se zpřístupní po prvním úspěšném fitu. ":
        "The MPC did not provide an initial orbit, so it will first be determined from the astrometry; element constraints become available after the first successful fit.",
    "MPC nevrátil použitelné kartézské elementy dráhy.":
        "The MPC did not return usable Cartesian orbital elements.",
    "MPC orbitální dotaz selhal: {0}":
        "MPC orbit query failed: {0}",
    "MPC stanice {0} není v katalogu.":
        "MPC station {0} is not in the catalogue.",
    "MPC stanice:":
        "MPC station:",
    "MPC vrátil nečitelnou odpověď: {0}":
        "The MPC returned an unreadable response: {0}",
    "MPC výchozí dráhu nemá; určuji počáteční stav přímo z astrometrie…":
        "The MPC has no initial orbit; determining the initial state directly from the astrometry…",
    "Mag / pásmo":
        "Mag / band",
    "Marsdenův vodní model (komety)":
        "Marsden water-ice model (comets)",
    "Merkur":
        "Mercury",
    "Metoda:":
        "Method:",
    "Mocninný model r⁻ᵐ (planetky / Yarkovsky)":
        "Power law r⁻ᵐ (asteroids / Yarkovsky)",
    "Měsíc":
        "Moon",
    "Měřítka variačních parametrů mají chybný rozměr.":
        "The scales of the variational parameters have a wrong size.",
    "N-body model planet ({0}); nejistota 1/a se nezobrazuje, protože kovariance epochy nebyla propagována přes planetární oblast až na hranici; nejde o publikovanou hodnotu MPC/JPL":
        "N-body model of the planets ({0}); the 1/a uncertainty is not shown because the epoch covariance was not propagated through the planetary region to the boundary; this is not a published MPC/JPL value",
    "NAVÁZÁNO {0}/{1} návratů ({2})":
        "LINKED {0}/{1} returns ({2})",
    "Najde a označí nejhorší dosud nevyřazené pozorování podle většího z absolutních reziduí RA a Dec.":
        "Finds and marks the worst observation not yet excluded, by the larger of the absolute RA and Dec residuals.",
    "Napětí modelu (nad {0:g}σ) v návratech ":
        "Model tension (above {0:g}σ) in returns",
    "Navázáno {0} z {1} návratů; ostatní jsou jen předpovědí.":
        "Linked {0} of {1} returns; the others are predictions only.",
    "Navázání pozorovaných návratů:":
        "Linkage of observed returns:",
    "Načte katalogovou dráhu; u NEOCP/PCCP stáhne astrometrii a spočítá vlastní dráhu.":
        "Loads the catalogue orbit; for NEOCP/PCCP it downloads the astrometry and computes an own orbit.",
    "Načte stav z MPC; u historických fragmentů Shoemaker–Levy 9 použije JPL Horizons.":
        "Loads the state from the MPC; for historical Shoemaker–Levy 9 fragments it uses JPL Horizons.",
    "Načteno {0} měření. Systematická metoda prohledá vzdálenost a radiální rychlost. Väisälova varianta vytvoří jednu hypotézu pro zadané q.":
        "{0} measurements loaded. The systematic method searches distance and radial velocity. The Väisälä variant creates one hypothesis for the given q.",
    "Načteno {0} pozorování{1}; počítám vlastní dráhu…":
        "Loaded {0} observations{1}; computing own orbit…",
    "Načteno {0} pozorování{1}. Výchozí dráha se tím nemění.":
        "Loaded {0} observations{1}. The initial orbit is not changed.",
    "Načtení MPC selhalo.":
        "Loading from the MPC failed.",
    "Načíst MPC pozorování":
        "Load MPC observations",
    "Načíst objekt":
        "Load object",
    "Načíst pozorování z TXT…":
        "Load observations from TXT…",
    "Načítám výchozí kartézský stav pro {0}…":
        "Loading the initial Cartesian state for {0}…",
    "Nebeský rovník":
        "Celestial equator",
    "Negravitační zrychlení":
        "Non-gravitational acceleration",
    "Nejdřív načtěte dráhu s kartézským stavem nebo spočítejte vlastní fit.":
        "First load an orbit with a Cartesian state or compute an own fit.",
    "Nejdřív vyberte alespoň jedno pozorování.":
        "First select at least one observation.",
    "Nejdřív zadejte označení objektu.":
        "First enter the object designation.",
    "Nejhorší nevyřazené pozorování: řádek {0}, {1}, reziduum {2:.2f}″.":
        "Worst observation not yet excluded: row {0}, {1}, residual {2:.2f}″.",
    "Nejhorší pozorování lze určit až po výpočtu reziduí.":
        "The worst observation can be determined only after the residuals are computed.",
    "Nejistota 1σ [″]":
        "Uncertainty 1σ [″]",
    "Nejprve načítám z MPC počáteční odhad; poté se vlastní fit spustí automaticky…":
        "First loading an initial estimate from the MPC; the own fit will then start automatically…",
    "Nelze určit směr perihelu.":
        "The perihelion direction cannot be determined.",
    "Nelze vytvořit soubor:\n{0}":
        "Cannot create the file:\n{0}",
    "Neplatná epocha. Použijte například 1900, 1900-01-01 nebo 1900-01-01 00:00.":
        "Invalid epoch. Use for example 1900, 1900-01-01 or 1900-01-01 00:00.",
    "Neplatná výstupní epocha.":
        "Invalid output epoch.",
    "Neplatná výstupní epocha. Použijte například 2026, 2026-10-18 nebo 2026-10-18 00:00.":
        "Invalid output epoch. Use for example 2026, 2026-10-18 or 2026-10-18 00:00.",
    "Neplatný parametr kuželosečky.":
        "Invalid conic parameter.",
    "Neplatný rozsah integrace.":
        "Invalid integration range.",
    "Neptun":
        "Neptune",
    "Neznámá metoda předběžné dráhy: {0}":
        "Unknown preliminary-orbit method: {0}",
    "Neznámá observatoř":
        "Unknown observatory",
    "Neznámý střed elementů: {0}":
        "Unknown element centre: {0}",
    "Neznámý způsob zadání rozsahu integrace.":
        "Unknown way of specifying the integration range.",
    "Nyní":
        "Now",
    "Návrat":
        "Return",
    "Návrat   Pozorovací oblouk  Dostupná  Ve fitu  RMS [\"]  RMS [σ]  Stav":
        "Return   Observed arc       Available  In fit  RMS [\"]  RMS [σ]  Status",
    "Návrat nebyl navázán.":
        "The return was not linked.",
    "Návrat nemá žádná použitelná měření.":
        "The return has no usable observations.",
    "Návrat {0}: hledám větev podél linie variací…":
        "Return {0}: searching for the branch along the line of variations…",
    "Návrat {0}: koriguji výchozí dráhu na jeho astrometrii…":
        "Return {0}: correcting the initial orbit to its astrometry…",
    "Návrat {0}: oblouk pokrývá celý oběh, uvolňuji {1}…":
        "Return {0}: the arc covers a full revolution, releasing {1}…",
    "Návrat {0}: předpovídám polohy z navázané dráhy…":
        "Return {0}: predicting positions from the linked orbit…",
    "Návrat {0}: společná korekce ({1})…":
        "Return {0}: joint correction ({1})…",
    "Návrat {0}: určuji samostatnou dráhu z jeho úhlů…":
        "Return {0}: determining an independent orbit from its own angles…",
    "Návrat {0}: {1}":
        "Return {0}: {1}",
    "Návraty byly navázány postupně od nejlépe pozorovaného ({0}) hledáním větve podél linie variací.":
        "Returns were linked one by one, starting from the best observed one ({0}), by searching for the branch along the line of variations.",
    "Název „{0}“ odpovídá více objektům:":
        "The name “{0}” matches several objects:",
    "ORBIT SOLVER — VÝVOJ DRÁHY":
        "ORBIT SOLVER — ORBIT EVOLUTION",
    "Objekt":
        "Object",
    "Objekt:\t{0}":
        "Object:\t{0}",
    "Obzorový (Az/Alt)":
        "Horizon (Az/Alt)",
    "Obzorový pohled vyžaduje pozemní stanici se známou polohou.":
        "The horizon view requires a ground station with a known position.",
    "Od:":
        "From:",
    "Odhad je orientační. Dlouhé integrační větve mohou trvat různě dlouho; zrušení se projeví po návratu právě běžící integrace.":
        "The estimate is approximate. Long integration branches may take varying time; cancelling takes effect after the running integration returns.",
    "Odlehlé měření při společném fitu návratů (> {0:g}σ od navázané dráhy).":
        "Outlying measurement in the joint fit of returns (> {0:g}σ from the linked orbit).",
    "Oficiální planetární kernel NASA/NAIF; pokrytí 1549-12-31 až 2650-01-25.":
        "Official NASA/NAIF planetary kernel; coverage 1549-12-31 to 2650-01-25.",
    "Oortův oblak; kandidát na dynamicky novou kometu":
        "Oort cloud; candidate dynamically new comet",
    "Orbit Solver 1.0\nAutor: Martin Mašek\nVyvíjeno s pomocí OpenAI Codex":
        "Orbit Solver 1.0\nAuthor: Martin Mašek\nDeveloped with the help of OpenAI Codex",
    "Orbit Solver narazil na chybu:\n\n{0}\n\nPodrobnosti: {1}":
        "Orbit Solver encountered an error:\n\n{0}\n\nDetails: {1}",
    "Orbit Solver — chyba":
        "Orbit Solver — error",
    "Orbit Solver — určování drah":
        "Orbit Solver — orbit determination",
    "Orbitální data MPC se nepodařilo načíst.":
        "The MPC orbit data could not be loaded.",
    "Orientační výpočet: planetární perturbace a zvolený model A1/A2/A3 jsou zahrnuty, polohy planet používají model {0}.":
        "Approximate calculation: planetary perturbations and the chosen A1/A2/A3 model are included; planet positions use the {0} model.",
    "Oskulační elementy jednotlivých návratů:":
        "Osculating elements of individual returns:",
    "Otevře dráhu ve zvětšeném samostatném okně. Klávesou F11 přepnete celou obrazovku.":
        "Opens the orbit in a larger separate window. Press F11 to toggle full screen.",
    "Otevře mapu ve zvětšeném samostatném okně. Klávesou F11 přepnete celou obrazovku.":
        "Opens the map in a larger separate window. Press F11 to toggle full screen.",
    "Ověřuji kontrolní součet staženého souboru…":
        "Verifying the checksum of the downloaded file…",
    "Ověřuji výchozí dráhu na všech návratech…":
        "Checking the initial orbit against all returns…",
    "P [roky]":
        "P [years]",
    "PERIHELIUM":
        "PERIHELION",
    "Perihelion: kalendářní datum mimo podporovaný rozsah [JD {0:.8f}]":
        "Perihelion: calendar date outside the supported range [JD {0:.8f}]",
    "Perihelium (UTC)":
        "Perihelion (UTC)",
    "Perihelium (UTC)      q [AU]       e           a [AU]      P [roky]    i [deg]    Node [deg]  Peri. [deg]":
        "Perihelion (UTC)      q [AU]       e           a [AU]      P [years]   i [deg]    Node [deg]  Peri. [deg]",
    "Perihelium ω [°]":
        "Perihelion ω [°]",
    "Perihelová vzdálenost q:":
        "Perihelion distance q:",
    "Planeta":
        "Planet",
    "Planetka":
        "Asteroid",
    "Planetocentrické elementy nelze určit: {0}":
        "Planetocentric elements cannot be determined: {0}",
    "Planetocentrické elementy vyžadují kartézský stav.":
        "Planetocentric elements require a Cartesian state.",
    "Planetární efemerida":
        "Planetary ephemeris",
    "Po fitu spočítat původní a budoucí barycentrickou dráhu v 250 AU":
        "After the fit, compute the original and future barycentric orbit at 250 AU",
    "Podle počtu let":
        "By number of years",
    "Podle počtu návratů":
        "By number of returns",
    "Pohled ze Země":
        "View from Earth",
    "Pohled:":
        "View:",
    "Pole RA a Dec nemají stejný rozměr.":
        "The RA and Dec arrays do not have the same size.",
    "Pole poloh pozorovatele neodpovídá epochám efemeridy.":
        "The observer position array does not match the ephemeris epochs.",
    "Polohovou nejistotu 1σ se nepodařilo dopočítat.":
        "The 1σ position uncertainty could not be computed.",
    "Polohovou nejistotu 1σ se nepodařilo určit.":
        "The 1σ position uncertainty could not be determined.",
    "Polohová nejistota 1σ byla dopočítána v úhlových vteřinách.":
        "The 1σ position uncertainty was computed in arcseconds.",
    "Popisek objektu":
        "Object label",
    "Popisky těles":
        "Body labels",
    "Poslední 2 roky":
        "Last 2 years",
    "Posledních 10 let":
        "Last 10 years",
    "Posledních 5 let":
        "Last 5 years",
    "Použita předběžná dráha. Efemerida je nastavena po 10 minutách na 48 hodin; výsledek může mít velkou a rychle rostoucí nejistotu.":
        "Preliminary orbit used. The ephemeris is set to 10-minute steps over 48 hours; the result may have a large and rapidly growing uncertainty.",
    "Použít":
        "Use",
    "Použít a zpřesnit":
        "Use and refine",
    "Použít předběžnou dráhu":
        "Use preliminary orbit",
    "Poznámka k fitu: {0}":
        "Fit note: {0}",
    "Poznámka: {0}":
        "Note: {0}",
    "Pozor: návraty jsou propojeny čistě gravitačním modelem; bez A1/A2/A3 mohou mít aktivní komety systematicky posunuté perihelové časy.":
        "Caution: the returns are linked by a purely gravitational model; without A1/A2/A3 active comets may have systematically shifted perihelion times.",
    "Pozorovací oblouk je příliš krátký pro samostatné počáteční určení dráhy.":
        "The observed arc is too short for an independent initial orbit determination.",
    "Pozorování":
        "Observations",
    "Počet návratů lze zadat jen pro uzavřenou eliptickou dráhu.":
        "A number of returns can be given only for a closed elliptic orbit.",
    "Počet souřadnic neodpovídá počtu časů pozorování.":
        "The number of coordinates does not match the number of observation times.",
    "Počet stanic neodpovídá počtu časů pozorování.":
        "The number of stations does not match the number of observation times.",
    "Počet časů neodpovídá počtu souřadnic.":
        "The number of times does not match the number of coordinates.",
    "Počáteční astrometrická dráha nevede na použitelnou kuželosečku.":
        "The initial astrometric orbit does not lead to a usable conic.",
    "Počáteční kombinaci A1/A2/A3/DT se nepodařilo stabilně integrovat. Zkuste začít s hodnotami parametrů blízkými nule.":
        "The initial combination of A1/A2/A3/DT could not be integrated stably. Try starting with parameter values close to zero.",
    "Počítám vlastní dráhu z astrometrie{0}{1}…":
        "Computing own orbit from the astrometry{0}{1}…",
    "Požadovaná epocha leží mimo rozsah JPL DE440 (1549-12-31 až 2650-01-25).":
        "The requested epoch lies outside the JPL DE440 range (1549-12-31 to 2650-01-25).",
    "Pro JPL DE440 chybí knihovna jplephem; nainstalujte requirements.txt.":
        "The jplephem library needed for JPL DE440 is missing; install requirements.txt.",
    "Pro Väisälovu dráhu zadejte kladnou perihelovou vzdálenost q.":
        "For a Väisälä orbit enter a positive perihelion distance q.",
    "Pro hledání větve chybí kovariance navázané dráhy.":
        "The covariance of the linked orbit needed for the branch search is missing.",
    "Pro kontrolu dráhy nejsou dostupná platná měření.":
        "No valid measurements are available for checking the orbit.",
    "Pro objekt „{0}“ nebyla nalezena pozorování ani v běžném MPC, ani na NEOCP/PCCP.":
        "No observations were found for the object “{0}” in the regular MPC nor on NEOCP/PCCP.",
    "Pro objekt „{0}“ nebyla v {1} nalezena pozorování.":
        "No observations were found for the object “{0}” in {1}.",
    "Pro předběžný výpočet jsou potřeba alespoň tři platná měření.":
        "At least three valid measurements are needed for the preliminary calculation.",
    "Pro přímé porovnání s publikovanou dráhou zapněte stejnou sadu parametrů. Například řešení JPL s A1+A2 nelze přesně porovnávat s fitem A1+A2+A3+DT; volné A3 a DT mohou kvůli korelacím posunout i A1 a A2.":
        "For a direct comparison with a published orbit, enable the same set of parameters. For example, a JPL solution with A1+A2 cannot be compared exactly with an A1+A2+A3+DT fit; free A3 and DT can also shift A1 and A2 through correlations.",
    "Pro toto počáteční určení dráhy je potřeba alespoň {0} platných měření.":
        "At least {0} valid measurements are needed for this initial orbit determination.",
    "Pro vlastní fit je potřeba alespoň šest platných astrometrických pozorování.":
        "The own fit needs at least six valid astrometric observations.",
    "Probíhá numerická integrace; u stovek let může výpočet trvat déle. Ostatní části programu zůstávají ovladatelné.":
        "Numerical integration in progress; for hundreds of years it may take longer. The rest of the program remains responsive.",
    "Prohledá možné vzdálenosti a radiální rychlosti, případně vytvoří Väisälovu hypotézu se zadanou perihelovou vzdáleností.":
        "Searches possible distances and radial velocities, or creates a Väisälä hypothesis with a given perihelion distance.",
    "Projekční plocha k ekliptice (3D)":
        "Projection surface to the ecliptic (3D)",
    "Provádím diferenciální korekci dráhy…":
        "Performing the differential correction of the orbit…",
    "Prověřuji kandidátní větev krátkého oblouku…":
        "Checking a candidate branch of the short arc…",
    "První hodnota je nevážené RMS v obloukových sekundách; druhá zohledňuje nejistoty rmsRA/rmsDec uvedené v ADES a částečnou korelaci série ze stejné stanice během jedné noci.":
        "The first value is the unweighted RMS in arcseconds; the second takes into account the rmsRA/rmsDec uncertainties given in ADES and the partial correlation of a series from the same station during one night.",
    "PŘEDBĚŽNÉ ŘEŠENÍ — DRÁHA NENÍ JEDNOZNAČNĚ URČENA\n\nMetoda: {0}\nEpocha: JD {1:.6f}\na = {2:.8g} AU\ne = {3:.8f}\nq = {4:.8f} AU\ni = {5:.6f}°\nΩ = {6:.6f}°\nω = {7:.6f}°\nM = {8:.6f}°\n\nVolba „Použít a zpřesnit“ spustí běžnou diferenciální korekci ze všech vybraných měření. Efemerida bude přednastavena po 10 minutách na 48 hodin.":
        "PRELIMINARY SOLUTION — THE ORBIT IS NOT UNIQUELY DETERMINED\n\nMethod: {0}\nEpoch: JD {1:.6f}\na = {2:.8g} AU\ne = {3:.8f}\nq = {4:.8f} AU\ni = {5:.6f}°\nΩ = {6:.6f}°\nω = {7:.6f}°\nM = {8:.6f}°\n\n“Use and refine” starts the regular differential correction from all selected measurements. The ephemeris will be preset to 10-minute steps over 48 hours.",
    "Předběžná dráha":
        "Preliminary orbit",
    "Předběžná dráha z krátkého oblouku":
        "Preliminary orbit from a short arc",
    "Předběžná dráha z krátkého oblouku • {0} • není jednoznačná":
        "Preliminary orbit from a short arc • {0} • not unique",
    "Předběžná dráha z krátkého oblouku…":
        "Preliminary orbit from a short arc…",
    "Předběžná dráha — výsledek":
        "Preliminary orbit — result",
    "Předběžné řešení nebylo použito.":
        "The preliminary solution was not used.",
    "Předchozí návraty:":
        "Previous returns:",
    "Předpovídám zbývající návraty z navázané dráhy…":
        "Predicting the remaining returns from the linked orbit…",
    "Předpověď jasnosti — fotometrický exponent n = {0:.2f}":
        "Brightness prediction — photometric exponent n = {0:.2f}",
    "Přeinstalovat JPL DE440…":
        "Reinstall JPL DE440…",
    "Překreslit":
        "Redraw",
    "Přepočítávám dráhu po vyřazení odlehlých měření…":
        "Recomputing the orbit after rejecting outliers…",
    "Přesně parabolická dráha vyžaduje perihelový parametr.":
        "An exactly parabolic orbit requires the perihelion parameter.",
    "Převod epochy se nezdařil: {0}":
        "The epoch conversion failed: {0}",
    "Převod epochy vyžaduje kartézský stav načtené nebo vypočtené dráhy.":
        "Converting the epoch requires the Cartesian state of a loaded or computed orbit.",
    "Převod používá N-body integraci s planetami a případnými A1–A3/DT. Aktivní planetární efemerida: {0}.":
        "The conversion uses N-body integration with the planets and any A1–A3/DT. Active planetary ephemeris: {0}.",
    "Převést na epochu":
        "Convert to epoch",
    "Přiblížení":
        "Zoom",
    "Připojuji předpovězené návraty {0} najednou…":
        "Joining the predicted returns {0} at once…",
    "Připraveno — zadejte objekt; jeho výchozí dráha se načte z MPC.":
        "Ready — enter an object; its initial orbit will be loaded from the MPC.",
    "Připravuji astrometrii a počáteční dráhu…":
        "Preparing the astrometry and the initial orbit…",
    "Připravuji výpočet…":
        "Preparing the calculation…",
    "Původní a budoucí barycentrická dráha:":
        "Original and future barycentric orbit:",
    "RA astrom. J2000 (h m s)":
        "RA astrom. J2000 (h m s)",
    "RA — východ vlevo":
        "RA — east to the left",
    "Rok návratu":
        "Return year",
    "Roky do budoucnosti:":
        "Years into the future:",
    "Roky do minulosti:":
        "Years into the past:",
    "Rovníkový (RA/Dec)":
        "Equatorial (RA/Dec)",
    "Rovníkový pohled (RA/Dec)":
        "Equatorial view (RA/Dec)",
    "Rozsah UTC:\t{0}\t{1}":
        "UTC range:\t{0}\t{1}",
    "Rozsah astrometrie:":
        "Astrometry range:",
    "Rozsah integrace musí být větší než nula.":
        "The integration range must be greater than zero.",
    "Rozsah:":
        "Range:",
    "Ruším…":
        "Cancelling…",
    "SELČ":
        "CEST",
    "SETKÁNÍ\t{0}\t{1}\t\t\t\t\t\t\t\t\t{2:.9f}\t{3:.6f}\t{4:.5f}\t{5}":
        "ENCOUNTER\t{0}\t{1}\t\t\t\t\t\t\t\t\t{2:.9f}\t{3:.6f}\t{4:.5f}\t{5}",
    "SEČ":
        "CET",
    "SPOLEČNÝ MODEL NEPOTVRZEN — fit 1/{0} návratů":
        "JOINT MODEL NOT CONFIRMED — fit 1/{0} returns",
    "STAV":
        "STATE",
    "Satelitní pozorování na řádku {0} postrádá druhý řádek s polohou družice.":
        "The satellite observation on line {0} lacks the second line with the satellite position.",
    "Shora":
        "Top",
    "Sklon i [°]":
        "Inclination i [°]",
    "Slunce":
        "Sun",
    "Sluneční soustava":
        "Solar System",
    "Směr integrace musí být -1 nebo +1.":
        "Integration direction must be -1 or +1.",
    "Soubor DE440 nelze použít: {0}":
        "The DE440 file cannot be used: {0}",
    "Soubor se nepodařilo uložit:\n{0}":
        "The file could not be saved:\n{0}",
    "Souvislá vedlejší astrometrická stopa; možný fragment nebo sekundární jádro.":
        "Coherent secondary astrometric track; possibly a fragment or a secondary nucleus.",
    "Společný N-body fit, ale s napětím modelu: návrat je jako celek posunut ({0}); konstantní silový model patrně nevystihuje proměnlivou aktivitu komety.":
        "Joint N-body fit, but with model tension: the return is shifted as a whole ({0}); a constant force model probably does not describe the comet's variable activity.",
    "Společný N-body fit; návrat sedí na vlastní astrometrii (≤ {0:g}σ).":
        "Joint N-body fit; the return fits its own astrometry (≤ {0:g}σ).",
    "Společný dynamický model nepotvrdil všechny návraty":
        "The joint dynamical model did not confirm all returns",
    "Spustit integraci":
        "Run integration",
    "Stahuji JPL DE440: {0:.1f} z {1:.1f} MB\nCíl: {2}":
        "Downloading JPL DE440: {0:.1f} of {1:.1f} MB\nTarget: {2}",
    "Stahuji astrometrii pro {0}…":
        "Downloading astrometry for {0}…",
    "Stahuji oficiální kernel JPL DE440 z NASA/NAIF…\nCíl: {0}":
        "Downloading the official JPL DE440 kernel from NASA/NAIF…\nTarget: {0}",
    "Stanice":
        "Station",
    "Stanice MPC:":
        "MPC station:",
    "Stažení JPL DE440":
        "Download JPL DE440",
    "Stažení selhalo:\n{0}":
        "Download failed:\n{0}",
    "Stáhnout JPL DE440 (cca 114 MB)…":
        "Download JPL DE440 (about 114 MB)…",
    "Stáhnout a znovu ověřit JPL DE440 (přibližně 114 MB)?":
        "Download and re-verify JPL DE440 (about 114 MB)?",
    "Stáhnout oficiální JPL DE440 z NASA/NAIF?\n\nSoubor má přibližně 114 MB a bude uložen do složky orbit-data. Pokrytí efemeridy je 1549–2650.":
        "Download the official JPL DE440 from NASA/NAIF?\n\nThe file is about 114 MB and will be saved to the orbit-data folder. The ephemeris covers 1549–2650.",
    "Střed elementů: {0}":
        "Element centre: {0}",
    "Střed intervalu:":
        "Interval centre:",
    "Střed: ":
        "Centre:",
    "Střed: {0}":
        "Centre: {0}",
    "Střední anomálie M [°]":
        "Mean anomaly M [°]",
    "Středový azimutální":
        "Centred azimuthal",
    "Středový azimutální pohled":
        "Centred azimuthal view",
    "Světelná křivka":
        "Light curve",
    "Systematické hledání vzdálenosti (doporučeno)":
        "Systematic distance search (recommended)",
    "Systematické rangování nenašlo použitelnou větev dráhy.":
        "Systematic ranging found no usable orbit branch.",
    "TXT: čitelný přehled":
        "TXT: readable summary",
    "Tabulka je hotová; polohová nejistota 1σ se počítá na pozadí…":
        "The table is ready; the 1σ position uncertainty is being computed in the background…",
    "Text neobsahuje žádné platné MPC 80-col pozorování.":
        "The text contains no valid MPC 80-col observation.",
    "Textové soubory (*.txt *.obs);;Všechny soubory (*)":
        "Text files (*.txt *.obs);;All files (*)",
    "Textové soubory (*.txt);;Všechny soubory (*)":
        "Text files (*.txt);;All files (*)",
    "Typ\tDatum UTC\tFáze / návrat / planeta\tq [AU]\te\ta [AU]\tP [roky]\t1/a [AU^-1]\ti [deg]\tOmega [deg]\tomega [deg]\tVzdálenost [AU]\tHillovy poloměry\tv rel. [km/s]\tDelta(1/a) [AU^-1]":
        "Type\tDate UTC\tPhase / return / planet\tq [AU]\te\ta [AU]\tP [years]\t1/a [AU^-1]\ti [deg]\tOmega [deg]\tomega [deg]\tDistance [AU]\tHill radii\tv rel. [km/s]\tDelta(1/a) [AU^-1]",
    "Téměř parabolická dráha nemá stabilní parametrizaci pomocí a, e.":
        "A nearly parabolic orbit has no stable parametrisation by a, e.",
    "Těleso nemůže být ve středu Slunce.":
        "The body cannot be at the centre of the Sun.",
    "Těleso {0!r} není v poskytovateli DE440 podporováno.":
        "The body {0!r} is not supported by the DE440 provider.",
    "Ukázková dráha (neřešená)":
        "Sample orbit (not solved)",
    "Uložit TXT…":
        "Save TXT…",
    "Uložit vývoj dráhy":
        "Save orbit evolution",
    "Uplynulo: 0 s • zbývající čas zatím nelze odhadnout":
        "Elapsed: 0 s • remaining time cannot be estimated yet",
    "Uran":
        "Uranus",
    "Uvolnění všech požadovaných negravitačních parametrů některý návrat zhoršilo; ponechán silový model z navazování.":
        "Releasing all requested non-gravitational parameters worsened some return; the force model from the linkage was kept.",
    "Uzel Ω [°]":
        "Node Ω [°]",
    "V zadaném rozsahu nejsou dostupné stavy dráhy.":
        "No orbit states are available in the chosen range.",
    "Ve fitu":
        "In fit",
    "Velká poloosa a [AU]":
        "Semi-major axis a [AU]",
    "Venuše":
        "Venus",
    "Vlastní dráha dokončena; RMS {0:.2f} obloukové sekundy{1}{2}{3}.":
        "Own orbit finished; RMS {0:.2f} arcseconds{1}{2}{3}.",
    "Vlastní fit":
        "Own fit",
    "Vlastní nastavení fitu":
        "Custom fit settings",
    "Vlastní nastavení…":
        "Custom settings…",
    "Vlastní nastavení… ({0})":
        "Custom settings… ({0})",
    "Vlastní objekt":
        "Own object",
    "Vlastní výpočet dráhy selhal.":
        "The own orbit calculation failed.",
    "Vložit MPC 80-col pozorování":
        "Paste MPC 80-col observations",
    "Vložit MPC pozorování…":
        "Paste MPC observations…",
    "Vložte jeden nebo více řádků astrometrie ve formátu MPC 80-column…":
        "Paste one or more lines of astrometry in the MPC 80-column format…",
    "Volba je standardně vypnutá. Výpočet může trvat podstatně déle; proběhne na pozadí. Aktivní model planet: {0}.":
        "The option is off by default. The calculation may take considerably longer; it runs in the background. Active planetary model: {0}.",
    "Vpřed ▶":
        "Forward ▶",
    "Vracím do fitu měření, která na výslednou dráhu sedí…":
        "Returning measurements that fit the final orbit back into the fit…",
    "Vstup a výstup jsou oskulační stavy na hranicích zvoleného intervalu. Svislé značky v grafu ukazují planetární setkání. Dvojklikem zobrazíte stav ve Sluneční soustavě.":
        "Inbound and outbound are osculating states at the limits of the chosen interval. Vertical marks in the plot show planetary encounters. Double-click to show the state in the Solar System.",
    "Vybere a vyřadí všechna dosud zařazená pozorování, u nichž |O−C RA| nebo |O−C Dec| přesahuje 2 obloukové sekundy. Poté je potřeba dráhu přepočítat.":
        "Selects and excludes all currently included observations whose |O−C RA| or |O−C Dec| exceeds 2 arcseconds. The orbit then needs to be recomputed.",
    "Vyberte objekt":
        "Select an object",
    "Vygenerovat efemeridu":
        "Generate ephemeris",
    "Vypnutím se souřadnicové pole uzamkne v aktuálním směru a objekt se po něm bude pohybovat.":
        "When unchecked, the coordinate grid is locked in its current direction and the object moves across it.",
    "Vypočítat předběžnou dráhu":
        "Compute preliminary orbit",
    "Vytvořeno programem Orbit Solver 1.0 — Martin Mašek":
        "Created by Orbit Solver 1.0 — Martin Mašek",
    "Vyřadit rezidua > 2″":
        "Exclude residuals > 2″",
    "Vyřadit rezidua > 2″ ({0})":
        "Exclude residuals > 2″ ({0})",
    "Vyřadit vybraná z fitu":
        "Exclude selected from fit",
    "Vyřazeno":
        "Excluded",
    "Vyřazeno {0} pozorování s reziduem > 2″; nyní dráhu přepočítejte.":
        "Excluded {0} observations with a residual > 2″; now recompute the orbit.",
    "Vzdálenost [AU]":
        "Distance [AU]",
    "Vázaná dráha má afel jen {0:.1f} AU a hranice {1:g} AU proto není pro tuto kometu použitelná.":
        "The bound orbit has an aphelion of only {0:.1f} AU, so the {1:g} AU boundary cannot be used for this comet.",
    "Väisälova hypotéza":
        "Väisälä hypothesis",
    "Väisälova hypotéza, zadané q = {0:.4f} AU":
        "Väisälä hypothesis, given q = {0:.4f} AU",
    "Väisälovou metodou":
        "by the Väisälä method",
    "Väisälä — předpoklad průchodu perihelem":
        "Väisälä — assumes perihelion passage",
    "Výběr objektu byl zrušen.":
        "The object selection was cancelled.",
    "Výchozí dráha předpovídá všechny návraty; koriguji je společně…":
        "The initial orbit predicts all returns; correcting them together…",
    "Výchozí dráha předpověděla všechny návraty; byly korigovány společně.":
        "The initial orbit predicted all returns; they were corrected together.",
    "Výchozí dráha {0} načtena; spouštím vlastní fit z astrometrie…":
        "Initial orbit {0} loaded; starting own fit from the astrometry…",
    "Výchozí kartézský stav {0}{1}":
        "Initial Cartesian state {0}{1}",
    "Výchozí návrat {0} nelze spolehlivě napasovat ({1}).":
        "The initial return {0} cannot be fitted reliably ({1}).",
    "Výpočet drah z astrometrie MPC":
        "Orbit computation from MPC astrometry",
    "Výpočet dráhy":
        "Orbit calculation",
    "Výpočet dráhy byl zrušen uživatelem.":
        "The orbit calculation was cancelled by the user.",
    "Výpočet dráhy byl zrušen; předchozí řešení zůstalo zachováno.":
        "The orbit calculation was cancelled; the previous solution was kept.",
    "Výpočet předběžné dráhy selhal.":
        "The preliminary orbit calculation failed.",
    "Výpočet se nezdařil: {0}":
        "The calculation failed: {0}",
    "Výpočet se spouští pouze tlačítkem. Návrat 0 je perihelový průchod nejbližší epoše dráhy.":
        "The calculation starts only with the button. Return 0 is the perihelion passage nearest to the orbit epoch.",
    "Výpočet zpožděné aktivity nekonverguje.":
        "The delayed-activity calculation does not converge.",
    "Výpočet zpožděné heliocentrické vzdálenosti selhal.":
        "The delayed heliocentric distance calculation failed.",
    "Výsledek z krátkého oblouku není jednoznačně určená dráha. Použijte jej jako počáteční odhad a počítejte s rychle rostoucí nejistotou efemeridy.":
        "A short-arc result is not a uniquely determined orbit. Use it as an initial estimate and expect a rapidly growing ephemeris uncertainty.",
    "Výsledná dráha NENÍ společným řešením všech načtených návratů. To samo o sobě nevyvrací jejich identitu: příčinou může být proměnlivá kometární aktivita nebo nedostatečný silový model.\n\n":
        "The resulting orbit is NOT a joint solution of all loaded returns. That alone does not disprove their identity: the cause may be variable cometary activity or an insufficient force model.\n\n",
    "Výstupní epocha":
        "Output epoch",
    "Výstupní epocha po fitu:":
        "Output epoch after the fit:",
    "Výstupní epocha:":
        "Output epoch:",
    "Výstupní časy musejí být konečná čísla.":
        "Output times must be finite numbers.",
    "Vývoj dráhy":
        "Orbit evolution",
    "Vývoj dráhy dokončen: {0} návratů, {1} významných přiblížení, {2} planetocentrických zachycení.":
        "Orbit evolution finished: {0} returns, {1} significant approaches, {2} planetocentric captures.",
    "Vývoj dráhy vyžaduje kartézský stav z MPC nebo z vlastního fitu.":
        "Orbit evolution requires a Cartesian state from the MPC or from an own fit.",
    "Výška":
        "Altitude",
    "Výška Slunce":
        "Sun altitude",
    "Výška kamery":
        "Camera elevation",
    "Všech šest elementů je ve vlastním fitu volných.":
        "All six elements are free in the own fit.",
    "YYYY nebo YYYY-MM-DD HH:MM":
        "YYYY or YYYY-MM-DD HH:MM",
    "Z astrometrie se nepodařilo sestavit použitelný počáteční odhad dráhy.":
        "No usable initial orbit estimate could be built from the astrometry.",
    "Z ekliptiky":
        "From the ecliptic",
    "Z úhlů krátkého oblouku nelze sestavit předběžnou dráhu.":
        "No preliminary orbit can be built from the angles of the short arc.",
    "ZACHYCENÍ\t{0}\t{1}\t{2}\tminimum {3:.9f} AU\t{4:.5f} Hillova poloměru":
        "CAPTURE\t{0}\t{1}\t{2}\tminimum {3:.9f} AU\t{4:.5f} Hill radii",
    "Zadané q neprotíná pozorovaný směr. Pro tuto geometrii musí být q alespoň {0:.3f} AU.":
        "The given q does not intersect the observed direction. For this geometry q must be at least {0:.3f} AU.",
    "Zadejte označení objektu.":
        "Enter the object designation.",
    "Zadejte platný pozemní kód stanice MPC.\nGeocentrický kód 500 nemá místní obzor.":
        "Enter a valid ground-based MPC station code.\nThe geocentric code 500 has no local horizon.",
    "Zavřít":
        "Close",
    "Zařadit vybraná do fitu":
        "Include selected in fit",
    "Zařazeno":
        "Included",
    "Zaškrtnuto = měření je zahrnuto do dalšího fitu.":
        "Checked = the measurement is included in the next fit.",
    "Zdroj integrace: vlastní fit z {0} pozorování.":
        "Integration source: own fit from {0} observations.",
    "Zdroj integrace: výchozí kartézská dráha MPC.":
        "Integration source: initial Cartesian orbit from the MPC.",
    "Zdroj integrace: žádný použitelný kartézský stav ještě není načten.":
        "Integration source: no usable Cartesian state has been loaded yet.",
    "Zdánlivá magnituda":
        "Apparent magnitude",
    "Země":
        "Earth",
    "Zkopírováno {0} pozorování včetně O−C RA a Dec.":
        "Copied {0} observations including O−C RA and Dec.",
    "Zkopíruje celé vybrané řádky včetně sexagesimální RA/Dec a obou reziduí O−C.":
        "Copies the whole selected rows including sexagesimal RA/Dec and both O−C residuals.",
    "Znovu načíst výchozí dráhu":
        "Reload initial orbit",
    "Zobrazen je oskulační návrat z Vývoje dráhy; načtená/fituovaná dráha zůstává beze změny.":
        "An osculating return from Orbit evolution is shown; the loaded/fitted orbit is unchanged.",
    "Zobrazena je epocha vlastního fitu. Převod používá N-body integraci; aktivní planetární efemerida: {0}.":
        "The epoch of the own fit is shown. The conversion uses N-body integration; active planetary ephemeris: {0}.",
    "Zobrazit dráhu":
        "Show orbit",
    "Zobrazí průsvitnou projekční plochu mezi dráhou tělesa a jejím kolmým průmětem do roviny ekliptiky.":
        "Shows a translucent projection surface between the body's orbit and its perpendicular projection onto the ecliptic plane.",
    "Zorné pole:":
        "Field of view:",
    "Zpřesňuji nejlepší větev na všech měřeních…":
        "Refining the best branch on all measurements…",
    "Zpřesňuji řešení na všech měřeních…":
        "Refining the solution on all measurements…",
    "Zrušit":
        "Cancel",
    "Zrušit výpočet":
        "Cancel calculation",
    "Zvětšit ↗":
        "Enlarge ↗",
    "Závislost na vzdálenosti:":
        "Distance dependence:",
    "Závěrečná korekce návrat nepotvrdila: ":
        "The final correction did not confirm the return:",
    "Závěrečná přesná korekce všech navázaných návratů…":
        "Final precise correction of all linked returns…",
    "analytický model":
        "analytic model",
    "ano":
        "yes",
    "argument perihelu ω [°]":
        "argument of perihelion ω [°]",
    "argumentu perihelu":
        "argument of perihelion",
    "astronomická noc":
        "astronomical night",
    "astronomický soumrak":
        "astronomical twilight",
    "azimut":
        "azimuth",
    "bez podrobností":
        "no details",
    "den":
        "day",
    "dokončuji poslední krok":
        "finishing the last step",
    "dráha dosud nenačtena":
        "orbit not loaded yet",
    "délka výstupného uzlu Ω [°]":
        "longitude of ascending node Ω [°]",
    "délky výstupného uzlu":
        "longitude of ascending node",
    "excentricita e":
        "eccentricity e",
    "fitovat":
        "fit",
    "fixovat":
        "fix",
    "g(r) = (1 AU / r)^m. Hodnota m = 2 je jednoduchý standardní model pro tlak záření nebo příčnou Yarkovského složku A2.":
        "g(r) = (1 AU / r)^m. The value m = 2 is a simple standard model for radiation pressure or the transverse Yarkovsky component A2.",
    "hrubý odhad zbývá: {0}":
        "rough estimate remaining: {0}",
    "jen integrovaná předpověď":
        "integrated prediction only",
    "jen samostatný fit":
        "independent fit only",
    "kometa":
        "comet",
    "krok {0} z nejvýše {1}":
        "step {0} of at most {1}",
    "krok {0} z {1}":
        "step {0} of {1}",
    "mocninný model m={0:g}":
        "power law m={0:g}",
    "např. 2026 nebo 2026-10-18 00:00":
        "e.g. 2026 or 2026-10-18 00:00",
    "nautický soumrak":
        "nautical twilight",
    "ne":
        "no",
    "nerozhodné — interval 3σ protíná parabolickou mez":
        "undecided — the 3σ interval crosses the parabolic limit",
    "neuvedeno":
        "not given",
    "neznámá chyba":
        "unknown error",
    "neznámá observatoř":
        "unknown observatory",
    "neznámá stanice":
        "unknown station",
    "nominálně mezihvězdná příchozí dráha":
        "nominally interstellar incoming orbit",
    "objekt ve středu":
        "object centred",
    "občanský soumrak":
        "civil twilight",
    "oběžná doba P [roky]":
        "orbital period P [years]",
    "oběžné doby":
        "orbital period",
    "okno {0:g} d: {1:.1f}σ":
        "window {0:g} d: {1:.1f}σ",
    "okno {0:g} d: {1}":
        "window {0:g} d: {1}",
    "perihel":
        "perihelion",
    "perihelová vzdálenost q [AU]":
        "perihelion distance q [AU]",
    "perihelové vzdálenosti":
        "perihelion distance",
    "pevné hvězdné pole":
        "fixed star field",
    "planetka":
        "asteroid",
    "pod obzorem":
        "below the horizon",
    "poloha není v katalogu MPC":
        "position not in the MPC catalogue",
    "počítám…":
        "computing…",
    "pravděpodobně opustí Sluneční soustavu":
        "will probably leave the Solar System",
    "propagace na návrat selhala: {0}":
        "propagation to the return failed: {0}",
    "předběžná dráha z krátkého oblouku: ":
        "preliminary orbit from the short arc:",
    "přímá korekce":
        "direct correction",
    "přímá předpověď: medián {0:.1f}σ":
        "direct prediction: median {0:.1f}σ",
    "příprava":
        "preparation",
    "přísluní\n{0:%Y-%m-%d %H:%M}":
        "perihelion\n{0:%Y-%m-%d %H:%M}",
    "r od Slunce":
        "r from Sun",
    "reciproké velké poloosy":
        "reciprocal semi-major axis",
    "rok":
        "year",
    "rozšíření oblouku se zastavilo na {0:.0f} z {1:.0f} dní":
        "arc extension stopped at {0:.0f} of {1:.0f} days",
    "samostatná dráha: {0}":
        "independent orbit: {0}",
    "sklon i [°]":
        "inclination i [°]",
    "sklonu dráhy":
        "inclination",
    "společný dynamický model neprošel":
        "joint dynamical model failed",
    "společný model nezkoušen":
        "joint model not attempted",
    "společně fitován":
        "jointly fitted",
    "systematicky":
        "systematically",
    "systematické hledání vzdálenosti a radiální rychlosti":
        "systematic search of distance and radial velocity",
    "systematický odhad":
        "systematic estimate",
    "v rel. [km/s]":
        "v rel. [km/s]",
    "vektor není ve výsledku":
        "vector not found in the result",
    "velká poloosa a [AU]":
        "semi-major axis a [AU]",
    "velké poloosy":
        "semi-major axis",
    "vstup":
        "inbound",
    "vyvoj-drahy":
        "orbit-evolution",
    "vázaná dráha; pravděpodobně již dříve prošla oblastí planet":
        "bound orbit; probably passed through the planetary region before",
    "výchozí dráha {0}":
        "initial orbit {0}",
    "výchozí dráha: {0}":
        "initial orbit: {0}",
    "výstup":
        "outbound",
    "výstupní epocha {0}":
        "output epoch {0}",
    "výška a elongace nejsou dostupné":
        "altitude and elongation are not available",
    "výška nad obzorem [°]":
        "altitude above the horizon [°]",
    "výška objektu {0:+.1f}°   •   výška Slunce {1:+.1f}°   •   elongace {2:.1f}°":
        "object altitude {0:+.1f}°   •   Sun altitude {1:+.1f}°   •   elongation {2:.1f}°",
    "zachycení: {0}":
        "capture: {0}",
    "zbývající čas zatím nelze odhadnout":
        "remaining time cannot be estimated yet",
    "zdánlivá magnituda":
        "apparent magnitude",
    "zůstává vázaná a měla by se vrátit":
        "remains bound and should return",
    "{0:%Y-%m-%d %H:%M UTC}    Δ od Země {1:.6f} AU    r od Slunce {2:.6f} AU{3}":
        "{0:%Y-%m-%d %H:%M UTC}    Δ from Earth {1:.6f} AU    r from Sun {2:.6f} AU{3}",
    "{0:%Y-%m-%d %H:%M} {1}    RA {2}    Dec {3}    mag {4:.2f}    Δ {5:.5f} AU    r {6:.5f} AU    elongace {7:.1f}°{8}":
        "{0:%Y-%m-%d %H:%M} {1}    RA {2}    Dec {3}    mag {4:.2f}    Δ {5:.5f} AU    r {6:.5f} AU    elongation {7:.1f}°{8}",
    "{0}  Krok vyhledávání událostí: {1:.2f} dne. {2}":
        "{0}  Event search step: {1:.2f} days. {2}",
    "{0}  Návrat 0 je perihelový průchod nejbližší epoše dráhy.  Tlačítkem Spustit integraci lze zvolit jiný rozsah.":
        "{0}  Return 0 is the perihelion passage nearest to the orbit epoch.  Use Run integration to choose a different range.",
    "{0} duplicitních redukcí téhož měření (jiný hvězdný katalog) bylo vynecháno z vah fitu.":
        "{0} duplicate reductions of the same measurement (different star catalogue) were left out of the fit weights.",
    "{0} odlehlých měření v navázaných návratech bylo automaticky vyřazeno.":
        "{0} outlying measurements in the linked returns were rejected automatically.",
    "{0} použitých":
        "{0} used",
    "{0} pozorování • {1}":
        "{0} observations • {1}",
    "{0} reprezentativních z {1} dostupných":
        "{0} representative of {1} available",
    "{0} reprezentativních z {1} v propojeném oblouku":
        "{0} representative of {1} in the linked arc",
    "{0} {1} vybraných pozorování.":
        "{0} {1} selected observations.",
    "{0} — dráha ve Sluneční soustavě":
        "{0} — orbit in the Solar System",
    "{0} — pohled ze Země":
        "{0} — view from the Earth",
    "{0} — vývoj {1} ({2}-centrické elementy)":
        "{0} — evolution of {1} ({2}-centric elements)",
    "{0} — {1}-centrický pohled — ":
        "{0} — {1}-centric view —",
    "{0} • Graf viditelnosti • {1} {2} • {3}\nstřed {4:%Y-%m-%d %H:%M} {5}":
        "{0} • Visibility plot • {1} {2} • {3}\ncentre {4:%Y-%m-%d %H:%M} {5}",
    "{0} • uplynulo: {1} • {2}":
        "{0} • elapsed: {1} • {2}",
    "{0} • {1} • {2} • trajektorie {3}":
        "{0} • {1} • {2} • trajectory {3}",
    "± 10 let":
        "± 10 years",
    "± 100 dní":
        "± 100 days",
    "± 150 dní":
        "± 150 days",
    "± 2,5 roku":
        "± 2.5 years",
    "± 200 dní":
        "± 200 days",
    "± 30 dní":
        "± 30 days",
    "± 300 dní":
        "± 300 days",
    "± 5 let":
        "± 5 years",
    "± 60 dní":
        "± 60 days",
    "± 600 dní":
        "± 600 days",
    "Úplná korekce kandidátních drah selhala.":
        "The full correction of the candidate orbits failed.",
    "úhlová vzdálenost k severu [°]":
        "angular distance to the north [°]",
    "úhlová vzdálenost — východ vlevo [°]":
        "angular distance — east to the left [°]",
    "Čas (UTC)":
        "Time (UTC)",
    "Čas (UTC)\tRA (h m s)\tDec (° ′ ″)\tMag / pásmo\tStanice\tPoužít\tO−C RA [″]\tO−C Dec [″]\tVe fitu":
        "Time (UTC)\tRA (h m s)\tDec (° ′ ″)\tMag / band\tStation\tUse\tO−C RA [″]\tO−C Dec [″]\tIn fit",
    "Čas UTC":
        "Time UTC",
    "Časové pásmo použité pro zadaný čas i popisky grafu.":
        "Time zone used for the entered time and the plot labels.",
    "Časový oblouk":
        "Time arc",
    "Časový výřez se počítá zpětně od nejnovějšího pozorování; data v tabulce zůstanou zachována.":
        "The time window is counted back from the newest observation; the data in the table are kept.",
    "Čekám na bezpečné ukončení právě probíhající integrace…":
        "Waiting for the running integration to finish safely…",
    "ČÁSTEČNÝ SPOLEČNÝ FIT {0}/{1} návratů ({2})":
        "PARTIAL JOINT FIT {0}/{1} returns ({2})",
    "Řádek {0} je poloha družice bez odpovídajícího řádku S s RA/Dec.":
        "Line {0} is a satellite position without a matching S line with RA/Dec.",
    "Řádek {0} není platná druhá část satelitního MPC pozorování.":
        "Line {0} is not a valid second part of a satellite MPC observation.",
    "Řádek {0} není platné optické MPC 80-col pozorování.":
        "Line {0} is not a valid optical MPC 80-col observation.",
    "Řádků: ":
        "Rows:",
    "Žádné perihelové průchody ani planetární setkání nebyly nalezeny.":
        "No perihelion passages or planetary encounters were found.",
    "Žádné zařazené pozorování nemá reziduum větší než 2″.":
        "No included observation has a residual larger than 2″.",
    "Žádný objekt":
        "No object",
    "Δ od Země":
        "Δ from Earth",
    "▶  Spočítat vlastní dráhu":
        "▶  Compute own orbit",
    "▶  Spočítat vlastní dráhu ({0} měření)":
        "▶  Compute own orbit ({0} measurements)",
    "▶  Spočítat vlastní dráhu ({0} z {1} měření)":
        "▶  Compute own orbit ({0} of {1} measurements)",
    "◀ Zpět":
        "◀ Back",
    "Načíst astrometrii z MPC":
        "Load astrometry from the MPC",
    "Použije astrometrii uloženou v místní cache, pokud už byla dříve stažena; jinak ji stáhne z MPC. Program sám vyzkouší běžný archiv MPC i aktuální NEOCP/PCCP.":
        "Uses the astrometry stored in the local cache if it was downloaded before; otherwise downloads it from the MPC. The program tries both the regular MPC archive and the current NEOCP/PCCP.",
    "Aktualizovat astrometrii z MPC":
        "Update astrometry from the MPC",
    "Znovu stáhne kompletní astrometrii z MPC (včetně nových pozorování) a přepíše místní cache.":
        "Downloads the complete astrometry from the MPC again (including new observations) and overwrites the local cache.",
    " z místní cache (staženo z MPC {0:%Y-%m-%d %H:%M} UTC; pro nová data použijte Aktualizovat astrometrii z MPC)":
        " from the local cache (downloaded from the MPC {0:%Y-%m-%d %H:%M} UTC; use Update astrometry from the MPC for new data)",
}
