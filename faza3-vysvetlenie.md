# Fáza 3 – Strojové učenie: kompletný sprievodca k nášmu notebooku

Tento dokument vysvetľuje časť **3. Strojové učenie** v `Foltin_Potancok_faza3.ipynb` (bunky `In [56]`–`In [75]`).
Je napísaný tak, aby sa z neho dalo učiť aj odpovedať na otázky cvičiaceho: čo ktorý pojem znamená,
prečo sme urobili konkrétne rozhodnutie a čo robí ktorý riadok kódu.

---

## Obsah

1. [Zadanie fázy 3 v skratke](#1-zadanie-fázy-3-v-skratke)
2. [Východiskový stav: s čím fáza 3 začína](#2-východiskový-stav-s-čím-fáza-3-začína)
3. [3.1 ID3 klasifikátor](#3-31-id3-klasifikátor)
4. [3.2 Rozhodovací strom a logistická regresia](#4-32-rozhodovací-strom-a-logistická-regresia)
5. [3.3 Hyperparameter tuning a ensemble](#5-33-hyperparameter-tuning-a-ensemble)
6. [3.4 Vplyv stratégie a nasadenie modelu](#6-34-vplyv-stratégie-a-nasadenie-modelu)
7. [Slovník pojmov](#7-slovník-pojmov)
8. [Otázky, ktoré môže cvičiaci položiť](#8-otázky-ktoré-môže-cvičiaci-položiť)
9. [Tabuľka čísel, ktoré treba vedieť](#9-tabuľka-čísel-ktoré-treba-vedieť)
10. [Ako si notebook znovu spustiť](#10-ako-si-notebook-znovu-spustiť)

---

## 1. Zadanie fázy 3 v skratke

Zadanie má štyri časti po 5 bodov:

| Časť | Čo sa vyžaduje | Kde to máme |
|---|---|---|
| **3.1** | Vlastný ID3 klasifikátor s hĺbkou aspoň 2 (vrátane koreňa), vyhodnotiť ho a vyriešiť overfitting | bunky `In [56]`–`In [60]` |
| **3.2** | Jeden stromový algoritmus zo scikit-learn, jeden nestromový, porovnať s ID3, vizualizovať pravidlá, vyhodnotiť metrikami | bunky `In [61]`–`In [65]` |
| **3.3** | Hyperparameter tuning, kombinácia modelov (ensemble), krížová validácia, dôkaz bez overfittingu | bunky `In [66]`–`In [71]` |
| **3.4** | Vyhodnotiť vplyv predspracovania, transformácií, výberu atribútov, vybrať najlepší model a opísať produkčnú pipeline | bunky `In [72]`–`In [75]` |

Kľúčová myšlienka celej fázy: **model, ktorý sa dobre učí, nie je to isté ako model, ktorý dobre predikuje**.
Preto sa celý čas porovnávajú dve čísla – výkon na trénovacej množine a výkon na množine, ktorú model nikdy nevidel.

---

## 2. Východiskový stav: s čím fáza 3 začína

Fáza 2 nám nechala v pamäti notebooku tieto premenné (fáza 3 ich priamo používa, nič znovu nenačítava):

| Premenná | Význam |
|---|---|
| `X_train`, `X_test` | 18 atribútov (15 spojitých + 3 kategorické) |
| `X_train_sel`, `X_test_sel` | 8 vybraných atribútov z časti 2.2 |
| `y_train`, `y_test` | cieľová premenná `failure` (0/1) |
| `build_pipeline(scaler_name, transformer_name)` | hotové predspracovanie z časti 2.3 |
| `seed = 20261023` | fixný seed pre reprodukovateľnosť |

**Rozdelenie dát**

- celkovo 10 089 záznamov,
- trénovacia množina **8 071** záznamov (80 %),
- testovacia množina **2 018** záznamov (20 %),
- podiel porúch: **8,76 %** v trénovacej množine.

To posledné číslo je najdôležitejšie pre celú fázu. Dáta sú **nevyvážené** (imbalanced):
na jednu poruchu pripadá približne 10 normálnych záznamov. Z toho vyplývajú dve veci, ktoré treba
mať stále na pamäti:

1. **Accuracy (presnosť) je zavádzajúca metrika.** Model, ktorý vždy povie "bez poruchy", má
   accuracy 91 % a je úplne nepoužiteľný. Preto používame F1-score triedy poruchy.
2. **Modely majú tendenciu predikovať majoritnú triedu.** Preto neskôr používame
   `class_weight="balanced"`, ktorý triedu poruchy pri učení zvýhodní.

**Osem vybraných atribútov** (`selected_features`):

```
service_hours, filter_dp, motor_current, discharge_dewpoint,
aux_current, working_pressure, reservoir_pressure, duty_class
```

**Predspracovanie z fázy 2** (`build_pipeline`) robí pre číselné atribúty:

```
QuantileTransformer(output_distribution="normal")  →  RobustScaler
```

a pre kategorické atribúty:

```
OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
```

`drop="first"` znamená, že z troch hodnôt `duty_class` (heavy, light, normal) vzniknú len dva
stĺpce – `duty_class_light` a `duty_class_normal`. Hodnota `heavy` je "vynechaná" a reprezentuje ju
stav, keď sú oba stĺpce nulové. Preto v strome neskôr vidíme podmienku `duty_class_light <= 0.50`,
čo v praxi znamená "vozidlo nie je ľahkej triedy".

---

## 3. 3.1 ID3 klasifikátor

### 3.1 A – Čo je ID3 a prečo ho píšeme sami

**ID3** (Iterative Dichotomiser 3, Quinlan, 1986) je historický algoritmus na stavbu rozhodovacieho
stromu. Zadanie ho vyžaduje preto, aby sme ukázali, že rozumieme **princípu** stromu a nie len
tomu, ako zavolať `DecisionTreeClassifier`.

Základná myšlienka: máme množinu záznamov, v ktorej sú premiešané obe triedy. Chceme ju rozdeliť
podľa jedného atribútu tak, aby v vzniknutých skupinách bola co najväčšia "čistota" – ideálne aby
každá skupina obsahovala len jednu triedu.

ID3 meria čistotu pomocou **entropie**:

$$H(S) = -\sum_i p_i \log_2 p_i$$

kde $p_i$ je podiel triedy $i$ v množine $S$.

- Ak sú všetky záznamy rovnakej triedy, $H = 0$ (žiadna neistota).
- Ak je pomer 50/50, $H = 1$ (maximálna neistota pre binárnu úlohu).

**Information gain** hovorí, o koľko sa entropia zníži, keď množinu rozdelíme podľa atribútu $A$:

$$IG(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

kde $S_v$ je podmnožina, v ktorej má atribút $A$ hodnotu $v$. Prvý člen je entropia pred rozdelením,
druhý je vážený priemer entropií po rozdelení (váhy sú podiely záznamov). ID3 v každom uzle vyberie
atribút s **najväčším** information gain.

### Discretizácia – prečo a ako

Pôvodný ID3 pracuje **iba s kategorickými atribútmi**. Naše senzory sú spojité (napr. `filter_dp`
od 7 do 117). Keby sme každú hodnotu brali ako samostatnú kategóriu, information gain by bol
nezmyselne vysoký a strom by sa naučil presné hodnoty naspamäť.

Riešenie: rozdelíme spojitý atribút na intervaly. V kóde (bunka `In [57]`):

```python
for col in num_sel:
    bins = pd.qcut(X_train_sel[col], q=5, retbins=True, duplicates="drop")[1]
    X_train_id3[col] = pd.cut(X_train_sel[col], bins=bins, include_lowest=True)
    X_test_id3[col] = pd.cut(X_test_sel[col], bins=bins, include_lowest=True)
```

Čo to robí:

- `pd.qcut(..., q=5, retbins=True)` vypočíta **kvantilové hranice** – 5 intervalov, v každom
  približne pätina záznamov (20 %, 40 %, 60 %, 80 % kvantil).
- `[1]` vyberie z výsledku len zoznam hraníc (`bins`).
- `pd.cut(..., bins=bins)` aplikuje **tie isté** hranice na trénovaciu aj testovaciu množinu.

Dve rozhodnutia, ktoré treba vedieť obhájiť:

1. **Prečo 5 intervalov a nie 3?** Pri 3 intervaloch strom nevidel dostatočný rozdiel medzi
   rizikovými a bezpečnými hodnotami a predikoval takmer vždy "bez poruchy" (F1 = 0). Jemnejšie
   delenie dá stromu viac možností. Toto je legitímna iterácia, zadanie ju priamo pripúšťa.
2. **Prečo hranice z trénovacej množiny?** Keby sme hranice počítali na všetkých dátach (alebo na
   testovacích), testovacia množina by ovplyvnila učenie – to sa volá **data leakage** (únik
   informácie). Hranice sú teda "naučený parameter" modelu, rovnako ako koeficienty regresie.

### Kód ID3 – bunka `In [56]` po častiach

**Entropia**

```python
def entropy(y):
    if len(y) == 0:
        return 0.0
    p = y.value_counts(normalize=True)
    return float(-np.sum(p * np.log2(p)))
```

`value_counts(normalize=True)` vráti podiely tried (napr. 0.91 a 0.09), z nich sa spočíta vzorec.
Prázdna množina má entropiu 0 (dohovor, aby nevzniklo delenie nulou).

**Information gain**

```python
def information_gain(X, y, column):
    total = entropy(y)
    weighted = 0.0
    for value, group in y.groupby(X[column], observed=True):
        weighted += len(group) / len(y) * entropy(group)
    return total - weighted
```

`y.groupby(X[column])` zoskupí cieľovú premennú podľa hodnôt jedného atribútu. `observed=True`
je potrebné pri kategorických dátach, aby pandas nebral všetky kombinácie kategórií, ale len tie,
ktoré sa naozaj vyskytujú. Cyklus spočíta vážený priemer entropií a odčíta ho od pôvodnej entropie.

**Strom ako slovník**

```python
node = {
    "prediction": counts.idxmax(),
    "proba": np.array([counts.get(c, 0.0) for c in self.classes_]),
    "n": len(y),
}
```

Uzol je obyčajný Python `dict`, nie vlastná trieda. Každý uzol si pamätá:

- `prediction` – väčšinovú triedu (použije sa, keď sa strom v tomto uzle zastaví),
- `proba` – podiel jednotlivých tried (potrebné pre ROC-AUC a `predict_proba`),
- `n` – počet záznamov v uzle (užitočné pri ladení).

Ak sa uzol delí, pribudnú mu kľúče `column` (podľa čoho delí) a `children` (slovník hodnota → podstrom).

**Koncové podmienky**

```python
if (depth >= self.max_depth
        or len(y) < self.min_samples_split
        or len(counts) == 1):
    return node
```

Strom sa zastaví, keď:

1. dosiahol maximálnu hĺbku (`depth >= max_depth`),
2. v uzle je príliš málo záznamov (`min_samples_split`),
3. všetky záznamy v uzle sú rovnakej triedy (uzol je čistý).

Navyše ak je najlepší gain nulový (`gains[best] <= 1e-9`), delenie nič neprinesie a strom skončí.

**Hĺbka stromu**

```python
def tree_depth(node):
    if "children" not in node:
        return 1
    return 1 + max(tree_depth(child) for child in node["children"].values())
```

List má hĺbku 1. Koreň má hĺbku 1 vtedy, keď je zároveň listom. Hĺbka 3 teda znamená koreň + dve
úrovne delenia. **Zadanie žiada aspoň 2 vrátane koreňa**; my trénujeme s hĺbkou 4 a neskôr 6.

**Predikcia**

```python
def _walk(self, X, values, predictions, node, mask):
    predictions[mask] = values(node)
    if "children" not in node:
        return
    column = X[node["column"]]
    for value, child in node["children"].items():
        child_mask = mask & (column == value).to_numpy()
        if child_mask.any():
            self._walk(X, values, predictions, child, child_mask)
```

Namiesto prechodu po jednom riadku (pomalé pri 8 000 riadkoch) prechádzame strom a pre každý uzol
zapíšeme jeho hodnotu do všetkých riadkov, ktoré doň patria. `mask` je pole `True/False`, ktoré
hovorí "tieto riadky sú v tomto uzle". Pre `predict` je hodnotou uzla `prediction`, pre
`predict_proba` je to `proba`.

`ID3Classifier` dedí z `ClassifierMixin, BaseEstimator` – vďaka tomu ho scikit-learn berie ako
klasifikátor a funguje v `cross_validate` rovnako ako `DecisionTreeClassifier`.

> **Bug, ktorý sme pri vývoji našli (dobré vedieť pri obhajobe).** Prvá verzia predikovala len
> "bez poruchy". Príčina bola v indexoch: po rozdelení sme volali `X[mask]` (pôvodné indexy)
> a `y[mask].reset_index(drop=True)` (nové indexy). Pandas pri `groupby` páruje podľa indexov, takže
> sa hodnoty zle priradili a information gain vyšiel nezmyselne. Riešenie: resetovať index na oboch
> stranách. Toto je klasická chyba pri práci s pandas a je fajn ju spomenúť.

### 3.1 B – Vyhodnotenie (bunka `In [58]`)

Používame päť metrík:

| Metrika | Čo hovorí | Prečo ju máme |
|---|---|---|
| **Accuracy** | podiel správnych predikcií | intuitívna, ale pri nevyvážených dátach zavádza |
| **Precision** | z tých, ktorým model povedal "porucha", koľko naozaj porúch bolo | koľko zbytočných kontrol vyrobíme |
| **Recall** (sensitivity) | z reálnych porúch, koľko model zachytil | koľko porúch nám ujde |
| **F1-score** | harmonický priemer precision a recall | hlavná metrika, vyvažuje obe |
| **ROC-AUC** | kvalita poradia pravdepodobností, nezávislá od prahu | používa sa pri porovnaní modelov |

Formálne:

$$P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F1 = \frac{2PR}{P + R}$$

`classification_report` vypíše tieto metriky pre obe triedy a navyše:

- **macro avg** – obyčajný priemer metrík oboch tried (obe triedy majú rovnakú váhu),
- **weighted avg** – priemer vážený počtom záznamov (dominuje majoritná trieda).

**Matica zámen** (confusion matrix) ukazuje konkrétne počty: koľko porúch sme zachytili, koľko sme
im povedali "v poriadku" a koľko normálnych záznamov sme omylom označili ako poruchu.

**Prvý model (hĺbka 4)** dopadol takto:

- accuracy 0,91 – na prvý pohľad výborne,
- recall triedy poruchy len **0,05** – zachytil 5 % porúch,
- F1 poruchy **0,08**.

Presne tu je vidieť, prečo accuracy klame. Model sa naučil "keď neviem, povedz bez poruchy"
a opticky vyzerá dobre.

### 3.1 C – Overfitting (bunky `In [59]`–`In [60]`)

**Overfitting (preučenie)** znamená, že model sa naučil konkrétne trénovacie záznamy (aj ich šum)
namiesto všeobecných pravidiel. Prejaví sa tak, že výkon na trénovacej množine ďalej rastie, ale na
novej množine stagnuje alebo klesá. Opak je **underfitting** – model je príliš jednoduchý a zle
funguje na oboch množinách.

**Ako to meriame:** trénujeme strom s postupne rastúcou hĺbkou a kreslíme dve krivky – F1 na
trénovacej a na testovacej množine (bunka `In [59]`). V tomto kroku zámerne používame
`min_samples_split=5`, aby sa efekt prejavil výrazne.

Nametené hodnoty:

| max_depth | Train F1 | Test F1 |
|---|---|---|
| 4 | 0,109 | 0,083 |
| 6 | 0,393 | 0,167 |
| 8 | 0,678 | 0,198 |
| 10 | 0,678 | 0,198 |

Trénovacia F1 stúpne z 0,39 na 0,68, ale testovacia sa za ten čas zlepší len z 0,17 na 0,20.
To je **nožnice** – typický obraz overfittingu.

Zaujímavý detail: do hĺbky 3 model predikuje iba "bez poruchy". Na to, aby strom našiel pravidlo pre
poruchu, potrebuje viac úrovní. To je zároveň dôvod, prečo zadanie žiada hĺbku aspoň 2 – plytký
strom sa vôbec nedostane k menšinovej triede.

**Riešenie overfittingu – dva kroky**

1. **Hĺbku volíme krížovou validáciou** (bunka `In [60]`), nie podľa testovacej množiny. Testovacia
   množina sa nesmie používať na rozhodovanie, inak prestáva byť nezávislým meradlom.
2. **Zvýšime `min_samples_split` z 5 na 20.** Uzol sa môže deliť len vtedy, keď obsahuje aspoň
   20 záznamov. Pravidlo tak nemôže vzniknúť z pár záznamov.

Vplyv druhej poistky pri hĺbke 6:

| min_samples_split | Train F1 | Test F1 |
|---|---|---|
| 5 | 0,393 | 0,167 |
| 20 | 0,274 | 0,175 |
| 50 | 0,147 | 0,127 |

Pri hodnote 20 klesne trénovacia F1 z 0,39 na 0,27 (model prestal memorovať) a testovacia sa mierne
zlepší. Pri 50 už klesá aj testovacia – model je primalý, nastáva **underfitting**. Preto volíme 20.

**Krížová validácia (cross-validation, CV)** – ako funguje:

1. Trénovaciu množinu rozdelíme na 5 častí (**foldov**).
2. Model natrénujeme 5-krát; vždy na 4 foldoch a testujeme na tom piatom.
3. Výsledok je priemer a smerodajná odchýlka piatich skóre.

Výhoda: každý záznam je raz vo validačnej množine, takže výsledok nezávisí od jedného náhodného
rozdelenia. `StratifiedKFold` navyše v každom folde zachová rovnaký podiel porúch.

**Výsledok ID3** (hĺbka 6, `min_samples_split=20`):

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| trénovacia | 0,919 | 0,626 | 0,175 | 0,274 | 0,901 |
| testovacia | 0,897 | 0,293 | 0,124 | 0,175 | 0,593 |

**Prečo je ID3 slabý?** Tri dôvody, ktoré treba vedieť povedať:

1. Diskretizácia zahodí informáciu – namiesto presnej hodnoty `filter_dp = 23,4` vidí len
   "je v štvrtej pätine".
2. ID3 delí podľa **celých kategórií** a nevie nájsť optimálny prah vnútri intervalu.
3. Nemá žiadnu regularizáciu okrem hĺbky, takže buď memoruje, alebo je primalý.

---

## 4. 3.2 Rozhodovací strom a logistická regresia

### 3.2 A – `DecisionTreeClassifier` (bunka `In [61]`)

scikit-learn implementuje **CART** (Classification and Regression Trees), nie ID3. Rozdiely:

| | ID3 (náš) | CART (sklearn) |
|---|---|---|
| vstup | len kategorické | spojité aj kategorické |
| delenie | podľa všetkých hodnôt atribútu | binárne podľa prahu `hodnota <= t` |
| kritérium | information gain (entropia) | Gini impurity (default) alebo entropia |
| počet vetiev | jedna na kategóriu | vždy dve |

**Gini impurity**:

$$Gini(S) = 1 - \sum_i p_i^2$$

Je to alternativa k entropii – meria pravdepodobnosť, že náhodne vybraný záznam bude zaradený zle.
V praxi dávajú obe kritériá veľmi podobné výsledky; v 3.3 skúšame obe.

Kľúčová výhoda CART pre nás: **prah hľadá priamo v spojitých hodnotách**, takže nepotrebuje
diskretizáciu a nestráca informáciu.

Model ale netrénujeme na "holých" dátach, ale v pipeline:

```python
tree_pipe = Pipeline([
    ("preprocessor", build_pipeline(scaler_name="robust", transformer_name="quantile")),
    ("model", DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=20,
        class_weight="balanced", random_state=seed)),
])
```

Čo znamenajú parametre:

- `max_depth=4` – maximálna hĺbka, poistka proti memorovaniu,
- `min_samples_leaf=20` – každý list musí obsahovať aspoň 20 záznamov (obdoba našej
  `min_samples_split`, ale vzťahuje sa na výsledný list),
- `class_weight="balanced"` – triedy dostanú váhy nepriamo úmerné svojej početnosti, takže
  menšinová trieda "váži" viac. Bez toho model takmer vždy predikuje nulu.
- `random_state=seed` – reprodukovateľnosť (strom má pri rovnakých dátach vždy rovnaký výsledok,
  ale pri niektorých nastaveniach je poradie náhodné).

**Výsledok:** accuracy 0,764, precision 0,181, recall 0,480, **F1 0,263**, ROC-AUC 0,652.

Oproti ID3 je to veľký rozdiel v recall (0,48 vs 0,12) – strom nájde takmer polovicu porúch.
Znížila sa tým accuracy, ale to je zámer: model, ktorý niečo zachytí, je užitočnejší ako model,
ktorý nič nezachytí a má peknú accuracy.

### 3.2 B – Logistická regresia (bunka `In [62]`)

**Prečo práve ona:** je to najjednoduchší a najčastejší "nestromový" klasifikátor, dobre
interpretovateľný a pri správnom predspracovaní spoľahlivý.

**Ako funguje:** spočíta vážený súčet atribútov a výsledok preženie sigmoidou:

$$z = w_0 + w_1x_1 + \dots + w_nx_n, \qquad p = \frac{1}{1 + e^{-z}}$$

`p` je pravdepodobnosť poruchy. Rozhodovacia hranica je **lineárna** – model nevie nájsť napríklad
pravidlo "porucha nastane, keď je filter_dp vysoký A ZÁROVEŇ service_hours vysoké", pokiaľ mu to
nevznikne z kombinácie atribútov. Presne preto je vhodná ako protiklad k stromu: ak dopadne podobne,
znamená to, že vzťahy v dátach nie sú výrazne nelineárne.

**Prečo potrebuje škálovanie:** regularizácia (default L2) trestá veľké koeficienty. Ak má
`service_hours` hodnoty v tisícoch a `duty_class` v jednotkách, koeficient pri `service_hours` musí
byť veľmi malý a optimalizácia je zle podmienená. Preto `RobustScaler` a `QuantileTransformer`
z fázy 2. (Podrobnejšie v 3.4.)

V kóde je `solver="liblinear"`. Je to voľba optimalizátora; pri neškálovaných dátach
(v experimentoch v 3.4) `lbfgs` nekonvergoval ani po 10 000 iteráciách a vypisoval
`ConvergenceWarning`, `liblinear` konverguje spoľahlivo.
`max_iter=5000` je poistka, `class_weight="balanced"` rovnaká logika ako pri strome.

**Výsledok:** accuracy 0,670, precision 0,157, recall 0,633, **F1 0,251**, ROC-AUC 0,694.

Zaujímavé je, že logistická regresia má najvyšší recall (0,63) ale najnižšiu precision – je
"opatrnejšia" a označí za poruchu viac záznamov. ROC-AUC má však najvyššie, čiže najlepšie
usporadúva pravdepodobnosti.

### 3.2 C a E – Porovnanie (bunka `In [63]`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| ID3 (hĺbka 6) | 0,897 | 0,293 | 0,124 | 0,175 | 0,593 |
| Decision Tree | 0,764 | 0,181 | 0,480 | **0,263** | 0,652 |
| Logistic Regression | 0,670 | 0,157 | 0,633 | 0,251 | **0,694** |

Ako to čítať (toto je dobré vedieť povedať nahlas):

- Poradie podľa F1: strom > logistická regresia > ID3.
- ID3 má najvyššiu accuracy, ale najhorší F1 – presne ten problém nevyvážených dát.
- Rozdiel medzi stromom a regresiou v ROC-AUC je malý (0,652 vs 0,694), takže ani jeden nie je
  jednoznačne lepší. Preto v 3.3 skúšame oba kombinovať.

### 3.2 D – Vizualizácia pravidiel (bunky `In [64]`–`In [65]`)

`plot_tree` nakreslí strom, `export_text` ho vypíše ako text. Ako čítať uzol:

```
|--- duty_class_light <= 0.50
|   |--- filter_dp <= 0.29
|   |   |--- motor_current <= 1.76
|   |   |   |--- class: 0
```

- Podmienka platí → ideme **doľava**, neplatí → doprava.
- `class: 0` znamená, že v danom liste prevažuje trieda "bez poruchy".
- Hodnoty `0.29`, `1.76` sú v jednotkách **po transformácii** (kvantilová transformácia + robustné
  škálovanie), preto nie sú v pôvodných jednotkách senzora. To treba pri obhajobe spomenúť, inak
  to vyzerá ako chyba.

`duty_class_light <= 0.50` znamená "vozidlo nie je ľahkej triedy" (one-hot s `drop="first"`).

Z pravidiel vidíme, že koreň delí podľa `duty_class` a ďalej sa objavujú `filter_dp`,
`motor_current`, `service_hours` a `working_pressure` – presne atribúty, ktoré vyšli ako najlepšie
v časti 2.2. To je pekná kontrola konzistencie medzi fázami.

Pravidlá sú krátke, pretože `max_depth=4` a `min_samples_leaf=20` nedovolia vytvoriť pravidlo pre
jedno konkrétne vozidlo – to je presne to, čo od modelu chceme.

---

## 5. 3.3 Hyperparameter tuning a ensemble

### 3.3 A – Čo sú hyperparametre a ako sa ladia

**Parameter** modelu je niečo, čo sa model naučí z dát (napr. prahy v stromoch, koeficienty
regresie). **Hyperparameter** je nastavenie, ktoré určujeme my pred učením (hĺbka stromu, počet
stromov, minimálny počet záznamov v liste).

**Grid search**: zadáme zoznam kandidátov pre každý hyperparameter a vyskúšame všetky kombinácie.
V našom prípade (bunka `In [66]`):

```python
param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_leaf": [1, 10, 30],
    "model__class_weight": [None, "balanced"],
}
```

`2 × 5 × 3 × 2 = 60` kombinácií, každá 5-krát v krížovej validácii → 300 trénovaní.

Dôležité detaily:

- Prefix `model__` hovorí, že parameter patrí kroku s názvom `model` v pipeline. Bez toho by
  `GridSearchCV` nevedel, kam parameter poslať.
- `scoring="f1"` – hodnotíme F1 triedy poruchy, nie accuracy (dôvod vysvetlený vyššie).
- `cv=5` – päťnásobná krížová validácia.
- `n_jobs=-1` – paralelne na všetkých jadrách.

**Najlepší výsledok mriežky:** gini, hĺbka 3, list 1, balanced → CV F1 **0,2291**.

### Prečo samotné "najlepšie CV F1" nestačí (bunka `In [68]`)

Pri Random Foreste mriežka vybrala hĺbku `None` (neobmedzená) s CV F1 0,3245. Problém: takýto les má
trénovaciu F1 0,82 a validačnú len 0,33. To je overfitting a zadanie výslovne žiada model bez neho.

Preto sme pridali druhé kritérium:

```python
cv_rf["gap"] = cv_rf["mean_train_score"] - cv_rf["mean_test_score"]
candidates = cv_rf[(cv_rf["mean_test_score"] >= best_score - best_std) & (cv_rf["gap"] <= 0.10)]
chosen = candidates.sort_values("mean_test_score", ascending=False).iloc[0]
```

Zjednodušene: **zoberieme len konfigurácie, ktoré nie sú viac ako jednu smerodajnú odchýlku od
najlepšej a zároveň nemajú rozdiel train–valid väčší ako 0,10, a z nich vyberieme tú s najvyšším
CV F1.**

Toto je obdoba známeho pravidla **"one standard error rule"** (vyber najjednoduchší model, ktorý je
štatisticky nerozoznateľný od najlepšieho). Hranica 0,10 je naša voľba – treba ju vedieť obhájiť:
"chceli sme, aby sa výkon na trénovacej množine neprenášal do modelu viac ako 10 percentuálnych
bodov F1; pri väčšom rozdiele už ide o memorovanie."

**Vybraná konfigurácia:** `max_depth=6`, `min_samples_leaf=20`, `class_weight="balanced"`,
300 stromov, CV F1 **0,2810**, gap **0,0991**.

Pre porovnanie, čo sme zamietli:

| Konfigurácia | CV F1 | gap (train−valid) |
|---|---|---|
| depth 6, leaf 20 | 0,2810 | 0,0991 |
| depth 6, leaf 50 | 0,2763 | 0,0664 |
| depth None, leaf 5 | 0,3245 | 0,4810 |

### 3.3 B – Ensemble: prečo Random Forest

**Ensemble** znamená kombinácia viacerých modelov. Idea: každý model robí iné chyby, a keď ich
skombinujeme, chyby sa čiastočne vyrušia.

**Random Forest** je **bagging** nad rozhodovacími stromami:

1. Vytvorí sa 300 stromov.
2. Každý strom sa učí na **náhodnej podmnožine záznamov vybranej s opakovaním** (bootstrap).
   Typicky dostane ~63 % unikátnych záznamov.
3. V každom uzle sa navyše vyberá len z náhodnej podmnožiny atribútov (pri klasifikácii typicky
   `sqrt(počet atribútov)`), aby stromy neboli navzájom rovnaké.
4. Predikcia je hlasovanie (väčšina) alebo priemer pravdepodobností.

Prečo to pomáha: jeden strom má **veľký rozptyl** – malá zmena v dátach zmení celý strom. Priemer
mnohých stromov má rozptyl menší, pričom vychýlenie (bias) zostáva podobné. Tento rozklad sa volá
**bias–variance tradeoff**.

**Soft voting** (bunka `In [69]`) kombinuje tri rôzne modely:

```python
voting = VotingClassifier(
    estimators=[("tree", tree_tuned), ("rf", rf_tuned), ("logreg", logreg_pipe)],
    voting="soft",
)
```

`soft` znamená, že sa priemerujú **pravdepodobnosti**, nie hlas tried. Je to citlivejšie ako
väčšinové hlasovanie (`hard`).

**Výsledky na testovacej množine:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (tuned) | 0,617 | 0,135 | 0,622 | 0,222 | 0,653 |
| Random Forest (tuned) | 0,743 | 0,174 | 0,514 | **0,260** | 0,681 |
| Voting (soft) | 0,708 | 0,166 | 0,582 | 0,259 | **0,688** |

Voting dopadol prakticky rovnako ako samostatný Random Forest. To je častý výsledok pri malých
dátach: ak sú modely natrénované na tých istých dátach, ich chyby nie sú nezávislé a kombinácia
neprinesie veľa.

> **Pozor na jednu vec, na ktorú sa môže cvičiaci spýtať:** "ladený" strom z 3.3 má na testovacej
> množine F1 0,222, teda horšie než strom z 3.2 s F1 0,263. Nie je to chyba. Grid search vybral
> hĺbku 3, pretože maximalizoval priemerné CV F1 na trénovacej množine; jednoduchší strom má menší
> rozptyl a v priemere cez foldy vyhrá, aj keď na tejto konkrétnej testovacej množine dopadne
> o čosi horšie. Presne preto sa rozhoduje podľa CV a nie podľa testu – inak by sme test použili na
> ladenie a prestal by byť nezávislý.

### 3.3 C a D – Dôkaz bez overfittingu (bunky `In [70]`–`In [71]`)

Používame `cross_validate(..., return_train_score=True)`. To je dôležitý trik: vráti nielen
validačné, ale aj **trénovacie** skóre pre každý fold, takže vidíme rozdiel priamo.

| Model | CV F1 (train) | Rozdiel train−valid |
|---|---|---|
| ID3 | 0,281 | 0,140 |
| Decision Tree (tuned) | 0,266 | 0,026 |
| Random Forest (tuned) | 0,380 | 0,094 |
| Logistic Regression | 0,274 | 0,003 |
| Voting (soft) | 0,308 | 0,036 |

Interpretácia:

- **Logistická regresia** má takmer nulový rozdiel – lineárny model s regularizáciou sa ťažko
  preučí. To ale nie je výhra, lebo má nízky výkon celkovo.
- **Decision Tree** má tiež malý rozdiel, ale nižší výkon než les.
- **Random Forest** má rozdiel 0,094 – pod našou hranicou 0,10 a najvyššie CV F1.
- **ID3** má najväčší rozdiel, čo potvrdzuje, že diskretizovaný strom memoruje najviac.

**Learning curve** (bunka `In [71]`) – čo ukazuje:

- os X: počet trénovacích záznamov (30 % až 100 % trénovacej množiny),
- os Y: F1,
- dve krivky: trénovacia a validačná, s pásom smerodajnej odchýlky medzi foldmi.

Ako sa číta:

| Čo vidíme | Diagnóza |
|---|---|
| obe krivky nízko a blízko seba | underfitting |
| veľká medzera medzi krivkami | overfitting |
| krivky sa zbiehajú, validačná ešte rastie | zdravý model, pomôže viac dát |

U nás sa validačná krivka s počtom dát mierne zlepšuje a trénovacia neletí opačným smerom, takže
model nemá underfitting a nemá ani klasický overfitting hlbokého stromu.

---

## 6. 3.4 Vplyv stratégie a nasadenie modelu

### 3.4 A a B – Vplyv predspracovania (bunky `In [72]`–`In [73]`)

Testujeme dve veci oddelene:

1. **Výber atribútov:** 18 atribútov vs 8 vybraných.
2. **Transformácie:** bez úprav, so scalerom, s transformátorom, rôzne kombinácie.

Pre každú kombináciu trénujeme rovnaký Random Forest s parametrami z 3.3 a meriame F1 a ROC-AUC na
testovacej množine. Kód je v bunkách `In [72]` a `In [73]`; pomocná funkcia `make_preprocessor` je zovšeobecnenie
`build_pipeline` na ľubovoľný zoznam atribútov (inak by sme museli meniť kód z fázy 2).

**Výsledok č. 1 – preprocessing pri stromoch nič nemení.**

Pre 8 vybraných atribútov vyšla F1 pre Random Forest vo všetkých kombináciách **0,2596** (ROC-AUC
0,6813–0,6814). Dôvod je matematický: strom delí podľa prahu `x <= t`. Ak na dáta použijeme
**monotónnu transformáciu** (škálovanie, kvantilová transformácia, power transformácia), poradie
hodnôt sa nezmení, len sa premenujú. Každý prah sa dá posunúť na nové hodnoty, takže strom nájde
presne tie isté delenia. Malé rozdiely v treťom desatinnom mieste vznikajú len zaokrúhľovaním
a zhlukovaním hodnôt po transformácii.

**Výsledok č. 2 – pri logistickej regresii je vplyv merateľný, ale malý.**

| Preprocessing (8 atribútov) | F1 (logistická regresia) |
|---|---|
| none + none | 0,2506 |
| robust + none | 0,2497 |
| robust + power | 0,2522 |
| robust + quantile | 0,2514 |

Rozdiely sú do 0,003. To znamená, že predspracovanie z fázy 2 nebolo "zázrak", ale bolo správne:
pomohlo hlavne **stabilite učenia** (bez škálovania konvergoval optimalizátor veľmi pomaly a pri
`lbfgs` ani nekonvergoval) a nie samotnému skóre. Toto treba povedať úprimne – v 2.1 D sme tvrdili,
že kombinácia scaleru a transformátora priniesla najlepšie výsledky; vo fáze 3 sa ukázalo, že na
tomto datasete je prínos malý. Je legitímne to takto dovysvetliť, zadanie priamo hovorí, že dáta
možno riešiť iteratívne a zmeny dokumentovať.

**Výsledok č. 3 – výber atribútov mierne pomohol.**

| Sada atribútov | CV F1 (priemer) | CV F1 (std) |
|---|---|---|
| všetkých 18 | 0,2800 | 0,0207 |
| vybraných 8 | **0,2854** | 0,0187 |

Rozdiel je malý, ale konzistentný a overený krížovou validáciou, nie jedným rozdelením. Praktický
prínos je hlavne v tom, že model má 8 vstupov namiesto 18 a je jednoduchší na údržbu. Atribúty
`fleet_code`, `model_code`, `cycle_month` a `cycle_dayofweek` s technickým stavom nesúvisia a v lese
zaberajú miesto v náhodných podmnožinách atribútov.

### 3.4 C – Vplyv metrík, algoritmov a ladenia

Zhrnutie, čo malo aký efekt:

| Zmena | Efekt |
|---|---|
| výber metriky (F1 namiesto accuracy) | zmenil výber modelu – inak by vyhral ID3 s 90 % accuracy |
| hĺbka stromu | najväčší efekt; neobmedzený strom mal gap 0,48, s hĺbkou 6 má 0,10 |
| `class_weight="balanced"` | zvýšil recall porúch, znížil accuracy |
| ensemble (RF, voting) | CV F1 +0,02 oproti jednému stromu, výrazne menší rozptyl |
| preprocessing | pri stromoch nulový, pri regresii malý |
| výber atribútov | +0,005 CV F1, výrazne jednoduchší model |

### 3.4 D – Ktorý model nasadiť

**Random Forest** s parametrami `max_depth=6`, `min_samples_leaf=20`, `class_weight="balanced"`,
300 stromov, na 8 vybraných atribútoch.

Argumenty (presne takto to treba povedať):

1. Spomedzi modelov bez overfittingu (gap ≤ 0,10) má **najvyššie CV F1** (0,281).
2. Nevyžaduje škálovanie – je odolný voči zmene rozsahu senzorov, takže ak sa v budúcnosti zmení
   jednotka merania, model sa nerozsype.
3. Voting dopadol rovnako, ale sú to tri modely na údržbu namiesto jedného.
4. Logistická regresia má nižší výkon a predpoklad linearity, ktorý tu neplatí.

**Testovacie metriky finálneho modelu** (z metadát):

| Metrika | Hodnota |
|---|---|
| Accuracy | 0,7428 |
| Precision | 0,1737 |
| Recall | 0,5141 |
| F1 | 0,2596 |
| ROC-AUC | 0,6813 |

Recall 0,51 znamená, že model zachytí približne polovicu porúch. Na reálne nasadenie to nie je
veľa, ale pri týchto dátach je to najlepšie, čo sme dosiahli, a je to čestne zdokumentované.

### 3.4 E – Produkčná pipeline

Pipeline v produkcii musí robiť **presne tie isté kroky ako pri trénovaní**, inak dostane iné
vstupy, než na aké je model zvyknutý (tzv. **training-serving skew**).

Naše kroky:

1. Načítať surové dáta v tvare `readings.csv`, `units.csv`, `gateways.csv`.
2. Spojiť podľa `unit_id` a vyčistiť podľa pravidiel fázy 1 (chýbajúce hodnoty, `motor_current = 99`,
   `oil_temperature = -999`).
3. Vybrať 8 atribútov.
4. Predspracovať: `QuantileTransformer` → `RobustScaler` pre číselné, one-hot pre `duty_class`.
5. Predikovať pravdepodobnosť a rozhodnúť podľa prahu.

Kroky 3 a 4 sú **zabalené v uloženom objekte**, takže v produkcii sa nedá zabudnúť na scaler:

```python
joblib.dump(final_model, "models/final_model.joblib")
```

`joblib` je knižnica na ukladanie Python objektov (podobne ako `pickle`, ale efektívnejšia pre numpy
polia). Uloží sa celý `Pipeline` vrátane natrénovaných transformátorov.

**Metadáta** – prečo ich ukladáme: model je binárny súbor, z ktorého po pol roku nikto nezistí, na
akých atribútoch a s akými metrikami vznikol. Preto `models/final_model_metadata.json` obsahuje:

| Kľúč | Význam |
|---|---|
| `model_type` | aký algoritmus |
| `trained_at` | kedy bol natrénovaný |
| `random_state` | seed, aby sa dal výsledok reprodukovať |
| `features` | presný zoznam a **poradie** atribútov |
| `preprocessing` | aké transformácie sú v pipeline |
| `hyperparameters` | zvolené hyperparametre |
| `training_rows`, `test_rows` | veľkosť dát |
| `failure_rate_train` | podiel porúch v trénovacej množine |
| `metrics_test` | dosiahnuté metriky |
| `decision_threshold` | prah rozhodnutia |
| `sklearn_version` | verzia knižnice (iná verzia môže načítať model inak) |

**Prah rozhodnutia a precision-recall krivka** (bunka `In [74]`):

```python
precision, recall, thresholds = precision_recall_curve(y_test, final_probs)
f1_curve = 2 * precision * recall / (precision + recall + 1e-12)
best_threshold = float(thresholds[np.argmax(f1_curve)])
```

Model štandardne rozhoduje pri prahu 0,5. Krivka ukázala, že pri prahu **0,63** by F1 bolo 0,3053
namiesto 0,2596. V praxi sa prah volí podľa toho, čo je drahšie:

- nižší prah → viac zachytených porúch, ale viac falošných poplachov,
- vyšší prah → menej poplachov, ale viac prehliadnutých porúch.

Preto je prah v metadátach a nie "zapečený" v modeli: prevádzka ho môže meniť bez pretrénovania.

> **Poznámka pre obhajobu:** prah 0,63 sme počítali na testovacej množine, čo je mierne optimistické.
> Správnejšie by bolo počítať ho na validačnej množine. Vedieť tento detail povedať je plus.

**Overenie načítania** (bunka `In [75]`): model načítame z disku, predikujeme rovnakých 8 záznamov a
porovnáme s pôvodnými predikciami. Zhoda je `True`, takže uloženie funguje a je reprodukovateľné.

---

## 7. Slovník pojmov

| Pojem | Vysvetlenie |
|---|---|
| **ID3** | Iterative Dichotomiser 3 – klasický algoritmus stavby stromu, delí podľa information gain, len kategorické atribúty |
| **CART** | Classification and Regression Trees – algoritmus v scikit-learn, binárne delenie podľa prahu, Gini alebo entropia |
| **Entropia** | miera neistoty/nečistoty množiny, 0 = čistá, 1 = 50/50 pri dvoch triedach |
| **Gini impurity** | alternativa entropie, $1 - \sum p_i^2$ |
| **Information gain** | pokles entropie po rozdelení podľa atribútu |
| **Koreň (root)** | prvý uzol stromu |
| **List (leaf)** | koncový uzol, ktorý už nedelí a vracia predikciu |
| **Hĺbka (depth)** | počet úrovní od koreňa po najhlbší list, koreň má hĺbku 1 |
| **Diskretizácia** | premena spojitého atribútu na intervaly/kategórie |
| **Kvantil** | hodnota, pod ktorou leží určitý podiel záznamov; 5 intervalov znamená hranice na 0, 20, 40, 60, 80 a 100. percentile |
| **Overfitting** | model memoruje trénovacie dáta, na nových zlyháva |
| **Underfitting** | model je primalý, zlyháva na trénovacích aj nových dátach |
| **Bias–variance tradeoff** | kompromis medzi jednoduchosťou (veľké vychýlenie) a citlivosťou na dáta (veľký rozptyl) |
| **Regularizácia** | penalizácia zložitosti modelu, aby sa nepreučil |
| **Train/test split** | rozdelenie dát na učenie a nezávislé testovanie |
| **Stratifikácia** | pri rozdelení zachová podiel tried v oboch častiach |
| **Krížová validácia (CV)** | rozdelenie trénovacej množiny na k foldov, k-krát trénovanie a validácia |
| **Fold** | jedna časť pri krížovej validácii |
| **Hyperparameter** | nastavenie určené pred učením (hĺbka, počet stromov...) |
| **Grid search** | systematické vyskúšanie všetkých kombinácií hyperparametrov |
| **One standard error rule** | vyber najjednoduchší model, ktorý je štatisticky nerozoznateľný od najlepšieho |
| **Ensemble** | kombinácia viacerých modelov |
| **Bagging** | učenie mnohých modelov na náhodných podmnožinách dát a ich spriemerovanie |
| **Bootstrap** | náhodný výber s opakovaním |
| **Random Forest** | bagging nad stromami + náhodný výber atribútov v uzloch |
| **Voting (soft/hard)** | kombinácia modelov podľa priemeru pravdepodobností / väčšinového hlasu |
| **Accuracy** | podiel správnych predikcií |
| **Precision** | presnosť pozitívnych predikcií, TP/(TP+FP) |
| **Recall** | záchyt pozitívov, TP/(TP+FN) |
| **F1-score** | harmonický priemer precision a recall |
| **ROC-AUC** | plocha pod ROC krivkou, kvalita usporiadania pravdepodobností, 0,5 = náhoda |
| **Confusion matrix** | tabuľka TP, FP, FN, TN |
| **Precision-recall krivka** | vzťah precision a recall pri rôznych prahoch |
| **Prah (threshold)** | hodnota pravdepodobnosti, od ktorej hovoríme "porucha" |
| **class_weight="balanced"** | váhy tried nepriamo úmerné ich početnosti |
| **One-hot encoding** | kategória → stĺpce 0/1; `drop="first"` vynechá jednu ako referenčnú |
| **Scaling** | zmena rozsahu hodnôt (Standard, Robust, MinMax) |
| **QuantileTransformer** | premena rozdelenia na rovnomerné alebo normálne |
| **PowerTransformer** | Yeo-Johnson transformácia približujúca rozdelenie normálnemu |
| **Monotónna transformácia** | nemení poradie hodnôt, preto nemení prahy stromu |
| **Pipeline** | reťazec krokov predspracovania a modelu, ktorý sa učí a používa ako celok |
| **ColumnTransformer** | aplikácia rôznych transformácií na rôzne stĺpce |
| **Data leakage** | informácia z testovacej množiny prenikne do učenia, výsledky sú potom falošne dobré |
| **Learning curve** | závislosť výkonu od množstva trénovacích dát |
| **joblib** | ukladanie/načítanie Python objektov |
| **Metadata** | sprievodné informácie o modeli (atribúty, metriky, verzia) |

---

## 8. Otázky, ktoré môže cvičiaci položiť

**Prečo ste implementovali ID3 sami, keď sklearn má hotový strom?**
Zadanie to vyžaduje (3.1 A). Navyše je vidieť rozdiel medzi ID3 a CART: diskretizácia a delenie
podľa kategórií vs prahy v spojitých hodnotách. Náš ID3 má F1 0,17, CART 0,26 – to je pekná ukážka,
čo znamená stratiť informáciu diskretizáciou.

**Prečo 5 intervalov pri diskretizácii?**
Pri 3 strom nenašiel žiadne pravidlo pre poruchu (F1 = 0). 5 je kompromis: dosť jemné na to, aby
signál existoval, dosť hrubé na to, aby v každom intervale bolo dosť záznamov (približne 1 600).

**Prečo sú hranice intervalov z trénovacej množiny?**
Aby nevznikol data leakage. Hranice sú naučený parameter, testovacia množina sa nesmie použiť pri
učení.

**Prečo meriate F1 a nie accuracy?**
Dáta sú nevyvážené (8,76 % porúch). Model, ktorý vždy povie "bez poruchy", má accuracy 91 % a F1 = 0.
F1 triedy poruchy je jediná metrika, ktorá núti model niečo zachytiť.

**Prečo je F1 taká nízka (0,26)?**
Pretože signál v dátach je slabý: poruchy sa v senzoroch prejavujú len málo a veľa záznamov je
nerozlíšiteľných. Ukazuje to aj learning curve – s viac dátami by sa model ešte zlepšoval. Model
však nie je pokazený: ROC-AUC 0,68 je lepšie ako náhoda (0,5) a recall 0,51 znamená, že polovicu
porúch zachytí.

**Ako ste zistili, že máte overfitting?**
Porovnali sme F1 na trénovacej a testovacej množine pri rastúcej hĺbke (bunka `In [59]`): trénovacia
stúpla z 0,39 na 0,68, testovacia len z 0,17 na 0,20. Neskôr sme to isté overili cez
`cross_validate(return_train_score=True)`.

**Prečo ste vybrali práve gap ≤ 0,10?**
Je to heuristická hranica: chceme, aby sa výkon na trénovacej množine neprenášal do modelu viac ako
10 percentuálnych bodov F1. Bez nej by mriežka vybrala neobmedzený les s CV F1 0,3245, ale s gap
0,48 – teda model, ktorý sa na testovacej množine správa o polovicu horšie.

**Prečo Random Forest a nie Voting, keď má voting lepšie ROC-AUC?**
Rozdiel v ROC-AUC je 0,007 (0,688 vs 0,681), čo je šum. Random Forest má lepšie F1 a je to jeden
model namiesto troch. Menej vecí, ktoré sa môžu v produkcii pokaziť.

**Prečo `max_depth=6` a `min_samples_leaf=20`?**
Vyšlo z krížovej validácie s podmienkou na gap. Hĺbka 6 je najhlbšia konfigurácia, ktorá ešte
neprekročí gap 0,10; 20 záznamov v liste zabráni pravidlám pre jednotlivé vozidlá.

**Prečo ste neškálovali dáta pre Random Forest?**
Stromy delia podľa prahu, a monotónna transformácia poradie hodnôt nemení, takže výsledok je
identický (overili sme to v 3.4: F1 0,2596 pre všetky kombinácie). Škálovanie sme nechali v pipeline
preto, že tam je aj kvôli logistickej regresii a kvôli konzistencii s fázou 2.

**Prečo je v strome podmienka `duty_class_light <= 0.50`?**
Pretože one-hot kódovanie má `drop="first"`: z troch tried vznikli dva stĺpce a `heavy` je
referenčná hodnota (oba stĺpce nulové). Podmienka teda znamená "vozidlo nie je ľahkej triedy".

**Prečo sú prahy v strome také malé čísla (0,29, 1,76)?**
Sú to hodnoty po `QuantileTransformer` a `RobustScaler`, nie pôvodné jednotky senzorov. Model ich
používa interne; pre interpretáciu sa musíme pozerať na poradie, nie na absolútne hodnoty.

**Prečo je v produkcii prah 0,5, keď 0,63 dáva lepšie F1?**
Prah je vec prevádzky, nie modelu. 0,5 je neutrálny default; ak je drahšia neplánovaná odstávka,
prevádzka prah zníži, ak sú drahé zbytočné kontroly, zvýši ho. Preto je prah v metadátach.

**Čo by ste zlepšili, keby ste mali viac času?**
Krátka a úprimná odpoveď: viac záznamov o poruchách (alebo SMOTE/undersampling), poradové
(prediktívne) atribúty ako "hodnota pred 7 dňami" namiesto jedného merania, prípadne gradient
boosting (LightGBM/XGBoost), ktorý na tabuľkových dátach zvyčajne porazí Random Forest. A prah by
sme ladili na validačnej množine, nie na testovacej.

**Kde je v kóde vidieť, že ste sa vyhli data leakage?**
Hranice diskretizácie, scaler, transformátor aj výber atribútov sa učia len na trénovacej množine
(`fit` na `X_train`), testovacia sa len transformuje (`transform`). Navyše pipeline drží všetko
pokope, takže sa to nedá omylom pomiešať.

---

## 9. Tabuľka čísel, ktoré treba vedieť

| Vec | Hodnota |
|---|---|
| záznamy celkom / train / test | 10 089 / 8 071 / 2 018 |
| podiel porúch | 8,76 % |
| vybraných atribútov | 8 z 18 |
| ID3: hĺbka po CV | 6 |
| ID3: F1 test / train | 0,175 / 0,274 |
| ID3: najhorší prejav overfittingu (hĺbka 8, mss 5) | train 0,678 vs test 0,198 |
| Decision Tree: F1 test | 0,263 |
| Logistická regresia: F1 test / ROC-AUC | 0,251 / 0,694 |
| Grid tree: kombinácií / najlepšie CV F1 | 60 / 0,229 |
| Random Forest: CV F1 / gap | 0,281 / 0,099 |
| Random Forest: F1 test | 0,260 |
| Voting: F1 test / ROC-AUC | 0,259 / 0,688 |
| finálny model: accuracy / precision / recall | 0,743 / 0,174 / 0,514 |
| finálny model: ROC-AUC | 0,681 |
| prah 0,63 → F1 | 0,305 |
| feature selection CV: 18 vs 8 atribútov | 0,280 vs 0,285 |
| súbory | `models/final_model.joblib`, `models/final_model_metadata.json` |

---

## 10. Ako si notebook znovu spustiť

```bash
uv run jupyter lab Foltin_Potancok_faza3.ipynb
```

alebo z terminálu:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace Foltin_Potancok_faza3.ipynb
```

Celý notebook sa spustí približne za 10–15 minút, najviac času berú krížové validácie ID3 (bunky `In [59]`
a `In [60]`) a grid search (bunky `In [66]` a `In [68]`). Spustenie vytvorí aj `data/processed_*.csv` (fáza 2) a
`models/*` (fáza 3).
