# Progetto di Lorenzo Menchini e Christian Petruzzella

## Setup e Installazione

1.  **Clona il Repository**:

2.  ```
    pip install -r requirements.txt
    ```
### 1. `env_testing.py`

* **Posizione**: `our_model/env_testing.py`
* **Modello Caricato Richiesto**: `our_model/agent_75_10k.pt`
* **Test**: Esegue `6` episodi di test.

### 2. `env_testing_v_rand.py`

* **Posizione**: `v_Start_noise/env_testing_v_rand.py`
* **Modello Caricato Richiesto**: `v_Start_noise/agent.pt`
* **Reset Ambiente**: L'auto dell'agente parte con una velocità iniziale randomizzata.
* **Test**: Esegue `4` episodi di test.

### 3. `env_testing_v_rand.py`

* **Posizione**: `v_Start_noise/env_testing_v_rand.py`
* **Modello Caricato Richiesto**: `v_Start_noise/agent.pt`
* **Reset Ambiente**: L'auto dell'agente parte con una velocità iniziale randomizzata.
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
* Per `physical.py`:
    ```bash
    cd v_Start_||/ 
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
