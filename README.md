# Progetto di Lorenzo Menchini e Christian Petruzzella

## Setup e Installazione

1.  **Clona il Repository**

2.  ```
    pip install -r requirements.txt
    cd final_models/
    ```

## script di test per:

## modello con velocità di partenza a 0
### 1. `env_testing.py`
* **Posizione**: `our_model/env_testing.py`

## modello con velocità di partenza randomizzata
### 2. `env_testing_v_rand.py`
* **Posizione**: `our_model_varying_speed/env_testing_v_rand.py`

## modello con caratteristiche fisiche aumentate
### 3. `env_testing_physical.py`

* **Posizione**: `original_physical_model/env_testing_physical.py`

## Come Eseguire gli Script

Gli script vengono eseguiti dalla riga di comando e accettano un argomento numerico che specifica lo scenario di test.

### Sintassi del Comando:

Prima, naviga nella directory appropriata:

* Per `env_testing.py`:
    ```bash
    cd final_models/our_model/
    ```
* Per `env_testing_v_rand.py`:
    ```bash
    cd final_models/our_model_varying_speed/ 
    ```
* Per `env_testing_physical.py`:
    ```bash
    cd final_models/original_physical_model/ 
    ```

Poi, esegui lo script:
```bash
python <nome_script.py> <env_type>
```

<env_type> è un numero intero da 1 a 4 che definisce lo scenario:
### 1: Scenario Facile: Jaywalker fermo, ostacolo (auto) distante.
### 2: Scenario Difficile: Jaywalker fermo, ostacolo (auto) vicino al jaywalker.
### 3: Scenario Molto Difficile: Jaywalker fermo, due auto ostacolo sulla corsia.
### 4: Iterazione Scenari: Esegue in sequenza gli scenari 1, 2, 3, ciclicamente per il numero di episodi di test.
