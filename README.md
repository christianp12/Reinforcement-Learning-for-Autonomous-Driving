# Progetto di Lorenzo Menchini e Christian Petruzzella

## Setup e Installazione

1.  **Clona il Repository**:
    Per prima cosa, clona questo repository sulla tua macchina locale (se non l'hai già fatto).

2.  **Installa le Dipendenze**:
    Assicurati di avere Python 3.x installato. Poi, naviga nella directory root del repository clonato ed esegui il seguente comando per installare tutte le librerie necessarie:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Struttura dei File Richiesti**:
    * Gli script si aspettano un'immagine per l'auto del giocatore. Assicurati che il file `carontop.png` sia presente in una directory `.car` alla radice del progetto: `/.car/carontop.png`.
    * I modelli pre-addestrati (`.pt`) devono trovarsi nelle rispettive directory degli script che li utilizzano (vedi sotto).

## Descrizione degli Script e Posizione

Entrambi gli script definiscono un ambiente di simulazione (`Jaywalker` class), un'architettura di rete neurale (`Lex_Q_Network`), e una classe `QAgent`.

### 1. `env_testing.py`

* **Posizione**: `our_model/env_testing.py`
* Questo script testa un modello pre-addestrato con condizioni iniziali dell'auto fisse.
* **Modello Caricato Richiesto**: `our_model/agent_75_10k.pt`
* **Reset Ambiente**: L'auto dell'agente parte da una posizione e velocità fisse.
* **Test**: Esegue `6` episodi di test.

### 2. `env_testing_v_rand.py`

* **Posizione**: `v_Start_noise/env_testing_v_rand.py`
* Questo script testa un modello pre-addestrato con una velocità iniziale dell'auto randomizzata e lievi differenze nel posizionamento degli ostacoli rispetto a Menco.
* **Modello Caricato Richiesto**: `v_Start_noise/agent.pt`
* **Reset Ambiente**: L'auto dell'agente parte con una velocità iniziale leggermente randomizzata.
* **Test**: Esegue `4` episodi di test.

## Come Eseguire gli Script

Gli script vengono eseguiti dalla riga di comando e accettano un argomento numerico che specifica lo scenario di test.

### Sintassi del Comando:

Prima, naviga nella directory appropriata:

* Per `env_testing_Menco.py`:
    ```bash
    cd our_model/
    ```
* Per `env_testing_v_rand.py`:
    ```bash
    cd v_Start_noise/ 
    ```

Poi, esegui lo script:
```bash
python <nome_script.py> <env_type>
```

<env_type> è un numero intero da 1 a 4 che definisce lo scenario:
1: Scenario Facile: Jaywalker fermo, ostacolo (auto) distante.
2: Scenario Difficile: Jaywalker fermo, ostacolo (auto) vicino al jaywalker.
3: Scenario Molto Difficile: Jaywalker fermo, due auto ostacolo sulla corsia.
4: Iterazione Scenari: Esegue in sequenza gli scenari 1, 2, 3, ciclicamente per il numero di episodi di test.
