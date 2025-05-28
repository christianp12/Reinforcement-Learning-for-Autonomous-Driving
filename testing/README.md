1.  **Collision (Collisione):**
    * **Descrizione:** Indica se l'auto ha avuto una collisione con il pedone o altri ostacoli.
    * **Ottimizzazione:** Evitare collisioni. Il valore ottimale è **0** (nessuna penalità). Una penalità (`reward[0] = -1`) viene assegnata in caso di collisione. Il modello dà alta priorità a mantenere il Q-value per questa metrica sopra una certa soglia (es. `>= -0.7`).

2.  **Forward (Avanzamento/Distanza):**
    * **Descrizione:** Misura il progresso dell'auto verso il traguardo.
    * **Ottimizzazione:** Massimizzare l'avanzamento. Il valore ottimale è un **valore positivo elevato**. Il reward (`reward[1]`) è proporzionale alla distanza percorsa in avanti e un grande bonus (`+100.0`) è dato al superamento del traguardo.

3.  **OOB (Out of Bounds/Fuori Corsia/Allineamento):**
    * **Descrizione:** Penalizza l'uscita dai limiti della strada e incentiva il mantenimento della corsia corretta (quella del traguardo).
    * **Ottimizzazione:** Evitare di uscire dai limiti della strada (penalità severa: `reward[2] -= 1000`) e rimanere nella corsia preferita. Essere nella corsia sbagliata dà una penalità (`reward[2] = -0.5`), mentre rimanere in quella corretta fornisce un piccolo reward crescente (`self.time_in_lane * 0.001`). Il valore ottimale è un **piccolo valore positivo crescente**, senza incorrere nelle penalità.

4.  **Completed (Completato):**
    * **Descrizione:** Indica se l'episodio è terminato con il raggiungimento del traguardo senza incidenti.
    * **Ottimizzazione:** Dopo ogni episodio, il valore booleano completed (risultato di env.step()) viene aggiunto a questa lista: self.completed.append(completed).
    Per la visualizzazione nell'interfaccia qqdm (la barra di progresso), viene calcolata la media degli ultimi 31 valori di completed: compl_mean = np.mean(self.completed[-31:]).

## Ordine di Considerazione delle Metriche (Approccio Lessicografico)

L'agente utilizza un Q-learning lessicografico, il che significa che considera le metriche (componenti del reward) in un ordine di priorità specifico durante la selezione dell'azione:

1.  **Priorità 1: Evitare Collisioni (`reward[0]`)**
    * L'agente cerca prima azioni che non portino a collisioni o che minimizzino la probabilità/gravità di una collisione.
2.  **Priorità 2: Massimizzare l'Avanzamento (`reward[1]`)**
    * Tra le azioni sicure, sceglie quelle che massimizzano il movimento in avanti verso il traguardo.
3.  **Priorità 3: Ottimizzare l'Allineamento/Evitare OOB (`reward[2]`)**
    * Infine, tra le azioni sicure e che massimizzano l'avanzamento, seleziona quelle che favoriscono il mantenimento della corsia corretta e l'evitamento dei bordi stradali.