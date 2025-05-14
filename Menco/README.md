README.md
Questo documento descrive gli aggiornamenti apportati alla classe Jaywalker nel nostro progetto di simulazione.

Novità implementate
Posizione iniziale casuale del pedone

Il pedone viene posizionato con coordinata X estratta casualmente nell’intervallo [dim_x/2, dim_x], ossia nella metà destra della strada.

La coordinata Y viene inizializzata a 0 o dim_y a seconda della direzione di attraversamento.

Movimento lungo l’asse Y

Il pedone si muove esclusivamente sull’asse Y.

In __init__ sono stati aggiunti gli attributi:
self.min_j_speed = 0.1
self.max_j_speed = 0.5
self.jaywalker_speed = 0.0
self.jaywalker_dir = 1
All’inizio del metodo step():


# Movimento pedone lungo Y
self.jaywalker[1] += self.jaywalker_speed * self.jaywalker_dir
if self.jaywalker[1] < 0 or self.jaywalker[1] > self.dim_y:
    # Nuova X casuale e inversione direzione
    self.jaywalker[0] = random.uniform(self.dim_x/2, self.dim_x)
    self.jaywalker_dir *= -1
    self.jaywalker[1] = np.clip(self.jaywalker[1], 0, self.dim_y)
# Aggiorna bounding box
self.jaywalker_max = self.jaywalker + self.jaywalker_r
self.jaywalker_min = self.jaywalker - self.jaywalker_r
Quando il pedone esce dai bordi verticali, rientra dall’altro lato con nuova X casuale e direzione invertita.

Rimozione del forcing di frenata

Sono state commentate tutte le righe di codice relative alla frenata obbligata (brake_start, brake_max e blocchi di calcolo del freno).

Il veicolo ora segue esclusivamente le azioni discrete selezionate dall’agente.

Dettaglio delle modifiche nei metodi
__init__
Aggiunti questi attributi per il pedone:

self.min_j_speed = 0.1
self.max_j_speed = 0.5
self.jaywalker_speed = 0.0
self.jaywalker_dir = 1
reset()
Il pedone viene inizializzato così:

# Posizione X casuale nella metà destra
self.jaywalker[0] = random.uniform(self.dim_x/2, self.dim_x)
# Direzione casuale
self.jaywalker_dir = random.choice([1, -1])
# Y all'estremità opposta
self.jaywalker[1] = 0 if self.jaywalker_dir == 1 else self.dim_y
# Velocità casuale
self.jaywalker_speed = random.uniform(self.min_j_speed, self.max_j_speed)
# Bounding box per collisioni
self.jaywalker_max = self.jaywalker + self.jaywalker_r
self.jaywalker_min = self.jaywalker - self.jaywalker_r
step(action)
Inserito all’inizio del metodo:

# Movimento pedone lungo Y
self.jaywalker[1] += self.jaywalker_speed * self.jaywalker_dir
if self.jaywalker[1] < 0 or self.jaywalker[1] > self.dim_y:
    # Nuova X casuale e inversione direzione
    self.jaywalker[0] = random.uniform(self.dim_x/2, self.dim_x)
    self.jaywalker_dir *= -1
    self.jaywalker[1] = np.clip(self.jaywalker[1], 0, self.dim_y)
# Aggiorna bounding box
self.jaywalker_max = self.jaywalker + self.jaywalker_r
self.jaywalker_min = self.jaywalker - self.jaywalker_r
Il resto del step() rimane invariato, tranne i blocchi di frenata che sono stati commentati.

Queste modifiche garantiscono che il pedone parta da una posizione casuale nella metà destra, si muova solo sull’asse Y a velocità casuale.

# rimossa completamente la fermata artificiale da noi programmata 