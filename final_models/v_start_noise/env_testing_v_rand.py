import numpy as np
from qqdm import qqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.image as mpimg

import os, time

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple

import random, math

import torch
from torch.linalg import solve as solve_matrix_system

import itertools
from multiprocessing import Pool, cpu_count

import sys

######## SET PARALLEL COMPUTING ##########
num_cores = cpu_count()

torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
##########################################


######## SET DEVICE ######################
# if torch.cuda.is_available():
#     device = "cuda:0"
# else:
#     device = "cpu"
device = "cpu"
#########################################


######### SET SEED ######################
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)
#########################################


################# SPEED UP ####################
torch.autograd.set_detect_anomaly(False);
torch.autograd.profiler.emit_nvtx(False);
torch.autograd.profiler.profile(False);
################################################


VectorScore = namedtuple('VectorScore', ('collision', 'distance', 'oob'))


# partiamo con azioni discrete
# reward = [collision, distance from target, centered in own lane]
# speed is saturated to 50 NO MORE

def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


class Car:
    def __init__(self, position):
        # distanze dal centro delle route anteriori e posteriori
        self.lf = 1
        self.lr = 1

        self.max_speed = 20.0
        #self.prev_position = np.zeros(2)

        self.reset(position)


    def move(self, df, a):
        #velocità
        self.v += a
        
        self.v = np.maximum(-self.max_speed, np.minimum(self.max_speed, self.v)) # speed saturation
        
        #phi e besta sono angoli di sterzata e angolo di sterzata effettivo
        #self.beta = np.arctan((1 + self.lr / self.lf) * np.tan(df))
        self.beta += np.arctan((1 + self.lr / self.lf) * np.tan(df))

        arg = self.phi + self.beta

        self.prev_position = np.copy(self.position)
        self.position += self.v * array([np.cos(arg), np.sin(arg)])

        #self.phi += self.v/self.lr * np.sin(self.beta)     ## Stearing inertia disabled
        #ang = array([np.cos(self.phi), np.sin(self.phi)])

        ang = array([np.cos(self.beta), np.sin(self.beta)])

        self.prev_front = self.front
        self.front = self.position + self.lf * ang

        #self.prev_back = self.back
        self.back = self.position - self.lr * ang

    # azzera la posizione del veicolo e le velocità
    def reset(self, position):
        self.position = position
        self.prev_position = np.copy(position)

        self.v = 0

        self.phi = 0
        self.beta = 0

        self.front = array([self.position[0] + self.lf, self.position[1]])
        self.back = array([self.position[0], self.position[1] + self.lr])

        self.prev_front = np.copy(self.front)
        #self.prev_back = np.copy(self.back)



class Jaywalker:

    def __init__(self):

        #modifiche per iterazione su scenari
        self.iter = False

        # MODIFICHE MOVIMENTO JAYWALKER
        self.min_j_speed = 0.1
        self.max_j_speed = 0.5
        self.jaywalker_speed = 0.0 
        self.jaywalker_dir = 1 
        
        #modifiche per curriculum
        self.curriculum_stage = 1

        #modifiche per rallebntamento avversario
        self.completed_mean = 0.0

        if not hasattr(self, 'car_img'):
            self.car_img = mpimg.imread("../.car/carontop.png")  # ← metti il file nella stessa cartella


        self.reward_size = 3

        self.dim_x = 120
        self.dim_y = 10

        #modifiche per aggiunta dinamica di ostacoli
        self.n_lanes = 2
        self.lane_width = self.dim_y / self.n_lanes
        self.lanes_y = [self.lane_width/2 + i * self.lane_width for i in range(self.n_lanes)]
        self.obstacles = []     # verrà popolato in reset()
        self.max_obstacles = 1  # ad es., fino a 5 oggetti

        self.goal = array([self.dim_x, self.dim_y/4])

        self.jaywalker = array([self.dim_x/2, self.dim_y/4])
        self.jaywalker_r = 2

        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        #self.df = [-np.pi/9, -np.pi/18, 0, np.pi/18, np.pi/9]
        self.df = [-np.pi/18, 0, np.pi/18]
        self.a = [-2, 0, 2]
        self.actions = [p for p in itertools.product(self.df, self.a)]

        self.num_df_actions = len(self.df)
        self.num_a_actions = len(self.a)

        self.state_size = 8
        self.action_size = self.num_df_actions * self.num_a_actions

        self.max_iterations = 15000

        self.car = Car(array([0.0,2.5]))

        self.counter_iterations = 0

        #self.prev_center_disance = np.abs(self.car.position[1] - self.goal[1])
        self.prev_target_distance = np.linalg.norm(self.car.front - self.goal)

        self.noise = 1e-5
        self.sight = 60
        self.sight_obstacle = 80

        self.scale_factor = 100

        # modifiche per grafico v
        self.velocity_history = []
        self.time_history = []
        self.current_time = 0
        self.max_time_display = 100  # Mostra gli ultimi 100 step temporali
        self.max_velocity_display = 25  # Scala massima per la velocità


    def collision_with_jaywalker(self):
        front = np.maximum(self.car.front, self.car.prev_front)
        prev = np.minimum(self.car.front, self.car.prev_front)

        denom = front - prev + self.noise

        upper = (self.jaywalker_max - prev) / denom
        lower = (self.jaywalker_min - prev) / denom

        scalar_upper = np.min(upper)
        scalar_lower = np.max(lower)
        
        if scalar_upper >= 0 and scalar_lower <= 1 and scalar_lower <= scalar_upper:
                return True
        
        return False


    def collision_with_obstacle(self):
        car_r = 2.0  # same as jaywalker radius
        car_front_max = self.car.front + car_r
        car_front_min = self.car.front - car_r

        car_prev_front_max = self.car.prev_front + car_r
        car_prev_front_min = self.car.prev_front - car_r

        for obs in self.obstacles:
            obs_max = obs['pos'] + obs['r']
            obs_min = obs['pos'] - obs['r']

            front = np.maximum(car_front_max, car_prev_front_max)
            prev = np.minimum(car_front_min, car_prev_front_min)

            denom = front - prev + self.noise

            upper = (obs_max - prev) / denom
            lower = (obs_min - prev) / denom

            scalar_upper = np.min(upper)
            scalar_lower = np.max(lower)

            if scalar_upper >= 0 and scalar_lower <= 1 and scalar_lower <= scalar_upper:
                return True

        return False


    # return the inverse of the distance from the jaywalker and the angle w.r.t to it
    def vision(self):
        vector_to_jaywalker = self.jaywalker - self.car.position + self.noise
        distance = np.linalg.norm(vector_to_jaywalker)

        if self.car.position[0] >= self.jaywalker[0] or distance > self.sight:
            return 0, -np.pi, float('inf')

        angle = np.arctan2(vector_to_jaywalker[1], vector_to_jaywalker[0])
        inv_distance = 1 / distance

        return inv_distance, angle, distance


    def vision_obstacle(self):
        if not self.obstacles:
            return 0, -np.pi

        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]

        vector_to_obs = obs['pos'] - self.car.position + self.noise
        distance = np.linalg.norm(vector_to_obs)

        if distance > self.sight_obstacle:
            return 0, -np.pi

        ref_angle = np.radians(15)  # inclined 15° left from forward
        angle_to_obs = np.arctan2(vector_to_obs[1], vector_to_obs[0])
        angle = angle_to_obs - ref_angle

        if np.abs(angle) > np.pi / 2:
            return 0, -np.pi  # outside 90° cone

        inv_distance = 1 / distance
        return inv_distance, angle



    def step(self, action):

        # MODIFICHE PER MOVIMENTO JAYWALKER
        self.jaywalker[1] += self.jaywalker_speed * self.jaywalker_dir
        # -- se il pedone esce dalla corsia sbuca da sotto  
        if self.jaywalker[1] < 0 or self.jaywalker[1] > self.dim_y:
            # riparti daccapo: nuova X e inverti direzione
            self.jaywalker[0] = random.uniform(self.dim_x/2, self.dim_x)
            self.jaywalker_dir *= -1
            # correggi Y all’interno
            self.jaywalker[1] = np.clip(self.jaywalker[1], 0, self.dim_y)
        # aggiorna bounding box del pedone
        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        if len(self.velocity_history) > self.max_time_display:
            self.velocity_history.pop(0)
            self.time_history.pop(0)


        #modifiche per aggiunta dinamica di ostacoli
        for obs in self.obstacles:
            obs['pos'][0] -= obs['v']  # si muovono verso sinistra

        df, a = self.actions[action]
        self.car.move(df, a) # default ripropaga l'accellerazione, altrimenti la modifico

        self.current_time += 1
        self.velocity_history.append(self.car.v)
        self.time_history.append(self.current_time)
        
        reward = np.zeros(self.reward_size)
        terminated = False
        completed = False

        if self.collision_with_jaywalker(): # collision with jaywalker
            reward[0] = -10
            terminated = True


        # distance from target
        reward[1] = (self.car.position[0] - self.car.prev_position[0])/self.scale_factor

        # accept surpassing the goal, terminate
        if self.car.front[0] >= self.goal[0]:
            if not terminated:
                completed = True
            terminated = True


        # collision with borders
        if self.car.front[1] > self.dim_y or self.car.front[1] < 0 or self.car.back[1] > self.dim_y or self.car.back[1] < 0 or self.car.position[0] < 0 or self.car.front[0] < 0:
            reward[2] -= 1000
            terminated = True 
        # distance from center of own lane
        else:
            # computes a distance-based penalty to encourage the car to stay centered in its lane
            reward[2] = -np.abs(self.car.position[1] - self.goal[1])

        reward[2] /= self.scale_factor * 10


        inv_distance, angle, jaywalker_distance = self.vision()

        # Find closest object
        obs_distance = float('inf')
        if self.obstacles:
            obs_distances = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
            obs_distance = min(obs_distances)
        
        # Call  vision_obstacle if jaywalker is detected or closest obstacle is closer than jaywalker
        if jaywalker_distance < float('inf') or obs_distance < jaywalker_distance:
            inv_distance_obs, angle_obs = self.vision_obstacle()
        else:
            inv_distance_obs, angle_obs = 0, -np.pi


        
        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]
        inv_d_obs = 1/dists[i_min]
        angle_obs = np.arctan((obs['pos'][1]-self.car.position[1])/(obs['pos'][0]-self.car.position[0]+self.noise))
        # corsia attuale (indice)
        lane_idx = min(range(len(self.lanes_y)), key=lambda i: abs(self.car.position[1]-self.lanes_y[i]))
        # nuovo state
        state = array([
            self.car.position[1],
            inv_distance, angle,          # Jaywalker info
            self.car.v, self.car.beta,
            inv_distance_obs, angle_obs, # Obstacle info
            float(lane_idx)
        ])

        self.counter_iterations += 1
        truncated = False

        if self.counter_iterations >= self.max_iterations:
            truncated = True

        if self.collision_with_obstacle():
            reward[0] -= 10
            terminated = True

        return state, reward, terminated, truncated, completed


    def reset(self):

        # reset dati graficali
        self.velocity_history = []
        self.time_history = []
        self.current_time = 0
        
        self.obstacles = []


        self.car.reset(array([0.0, 2.5]))
        self.counter_iterations = 0

        # agent random start velocity implementation
        sigma = 1.5
        v_base = 2.0
        v_final = (v_base + np.random.normal(0, sigma))
        v_final = max(0.0, v_final) # evito v negative
        self.car.v = v_final


        # --- Pedone fermo a metà strada, posizione fissa ---
        self.jaywalker = array([self.dim_x * 0.5, self.dim_y / 4])
        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        lane = self.lanes_y[1]

        # # --- Scenario 1: ostacolo distante (sorpasso possibile) ---
        if self.env_type == 1:
            pos_x = self.dim_x - 80  # lontano dal pedone
            speed = 0.5      # lento
            self.jaywalker_speed = 0.0
            self.jaywalker_dir = 0
            self.obstacles.append({ 
                'type': 'car',
                'pos': array([pos_x, lane]),
                'r': 2.0,
                'v': speed
            })

        # # # --- Scenario 2: ostacolo vicino (sorpasso critico) ---
        elif self.env_type == 2:
            pos_x = self.jaywalker[0] + 20  # Default position for critical scenario
            speed = 1     
            self.jaywalker_speed = 0.0
            self.jaywalker_dir = 0
            self.obstacles.append({ 
                'type': 'car',
                'pos': array([pos_x, lane]),
                'r': 2.0,
                'v': speed
            })           
        
        # # # --- Scenario 3: due ostacoli (sorpasso critico) ---
        elif self.env_type == 3:
            speed = 2 #4 
            #first car
            self.obstacles.append({ 
                'type': 'car',
                'pos': array([120, lane]),
                'r': 2.0,
                'v': 2
            })
            #second car
            self.obstacles.append({ 
                'type': 'car',
                'pos': array([80, lane]),
                'r': 2.0,
                'v': 2
            })

        # Stato iniziale
        inv_distance, angle, _ = self.vision()
        dists = [np.linalg.norm(obs['pos'] - self.car.position) for obs in self.obstacles]
        i_min = np.argmin(dists)
        obs = self.obstacles[i_min]
        inv_d_obs = 1 / dists[i_min]
        angle_obs = np.arctan((obs['pos'][1] - self.car.position[1]) / (obs['pos'][0] - self.car.position[0] + self.noise))
        lane_idx = min(range(len(self.lanes_y)), key=lambda i: abs(self.car.position[1] - self.lanes_y[i]))

        inv_distance_obs, angle_obs = 0.0, -np.pi

        # modifiche per iterazioni
        if self.iter:
            if self.env_type == 3:
                self.env_type = 0
            self.env_type += 1

        return array([
            self.car.position[1],
            inv_distance,
            angle,
            self.car.v,
            self.car.beta,
            inv_distance_obs,
            angle_obs,
            float(lane_idx)
        ])

    

    def random_action(self):
        return int(np.floor(random.random() * self.action_size))


    def __str__(self):
        return "jaywalker"
    
    def render(self):
        # Crea una figura con due subplot: strada sopra, grafico sotto
        if not hasattr(self, 'fig'):
        # Crea la figura solo la prima volta
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
            plt.ion()
            plt.show()
        
        ax1 = self.ax1
        ax2 = self.ax2
        # ========== SUBPLOT 1: VISUALIZZAZIONE STRADA ==========
        ax1.clear()
        
        # Sfondo strada nero
        road = mpatches.Rectangle((0, 0), self.dim_x, self.dim_y,
                                facecolor='black', edgecolor='none')
        ax1.add_patch(road)

        # Linea di arrivo verde tratteggiata
        gx = self.goal[0]
        ax1.plot([gx, gx], [0, self.dim_y],
                color='lime', linewidth=3,
                linestyle=(0, (5, 5)),
                label='Finish')

        # Linee di bordo continue bianche
        ax1.plot([0, self.dim_x], [0, 0], color='white', linewidth=2)
        ax1.plot([0, self.dim_x], [self.dim_y, self.dim_y], color='white', linewidth=2)

        # Linea centrale tratteggiata bianca
        mid_y = self.dim_y / 2
        ax1.plot([0, self.dim_x], [mid_y, mid_y],
                color='white', linewidth=1,
                linestyle=(0, (10, 10))) 

        # Pedone come cerchio rosso
        circle_j = plt.Circle(self.jaywalker, self.jaywalker_r, color='red', alpha=0.7)
        ax1.add_patch(circle_j)

        # Ostacoli (macchine arancioni)
        for obs in getattr(self, 'obstacles', []):
            c = 'orange' if obs['type'] == 'car' else 'green'
            circle_o = plt.Circle(obs['pos'], obs['r'], color=c, alpha=0.6)
            ax1.add_patch(circle_o)

        # Auto del giocatore
        car = self.car
        car_length = 3.0
        car_width = 3.0
        arg = car.phi + car.beta

        # Coordinate per posizionare l'immagine
        extent = [
            car.position[0] - car_length / 2,
            car.position[0] + car_length / 2,
            car.position[1] - car_width / 2,
            car.position[1] + car_width / 2
        ]

        # Trasformazione per ruotare l'immagine
        img_transform = transforms.Affine2D().rotate_around(
            car.position[0], car.position[1], arg
        ) + ax1.transData

        # Mostra immagine dell'auto
        ax1.imshow(self.car_img, extent=extent, transform=img_transform, zorder=5)
        
        ax1.set_xlim(-1, self.dim_x+1)
        ax1.set_ylim(-1, self.dim_y+1)
        ax1.set_title('Simulazione Guida Autonoma', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Posizione X')
        ax1.set_ylabel('Posizione Y')
        
        # ========== SUBPLOT 2: GRAFICO VELOCITÀ ==========
        ax2.clear()
        
        if len(self.velocity_history) > 1:
            # Crea il grafico della velocità con colori gradienti
            time_array = np.array(self.time_history)
            velocity_array = np.array(self.velocity_history)
            
            # Colori diversi in base alla velocità
            colors = []
            for v in velocity_array:
                if v < 5:
                    colors.append('#00ff00')  # Verde per velocità basse
                elif v < 15:
                    colors.append('#ffff00')  # Giallo per velocità medie
                else:
                    colors.append('#ff0000')  # Rosso per velocità alte
            
            # Disegna il grafico a linee
            ax2.plot(time_array, velocity_array, 
                    color='#1f77b4', linewidth=2.5, alpha=0.8)
            
            # Aggiungi punti colorati
            ax2.scatter(time_array, velocity_array, 
                    c=colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Riempi l'area sotto la curva con gradiente
            ax2.fill_between(time_array, velocity_array, alpha=0.3, 
                            color='#1f77b4', interpolate=True)
        
        # Configurazione assi del grafico
        if len(self.time_history) > 0:
            ax2.set_xlim(max(0, self.current_time - self.max_time_display), 
                        max(self.max_time_display, self.current_time))
        else:
            ax2.set_xlim(0, self.max_time_display)
        
        ax2.set_ylim(-self.max_velocity_display, self.max_velocity_display)
        
        # Linee di riferimento
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.axhline(y=self.car.max_speed, color='red', linestyle='--', alpha=0.5, label=f'Max Speed ({self.car.max_speed})')
        ax2.axhline(y=-self.car.max_speed, color='red', linestyle='--', alpha=0.5)
        
        # Griglia
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        # Etichette e titolo
        ax2.set_xlabel('Tempo (steps)', fontsize=12)
        ax2.set_ylabel('Velocità', fontsize=12)
        ax2.set_title(f'Velocità in Tempo Reale - Attuale: {self.car.v:.2f}', 
                    fontsize=12, fontweight='bold')
        
        # Aggiungi valore corrente come testo
        if len(self.velocity_history) > 0:
            current_v = self.velocity_history[-1]
            ax2.text(0.02, 0.95, f'V = {current_v:.2f}', 
                    transform=ax2.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Legenda per i colori della velocità
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ff00', 
                markersize=8, label='Bassa (< 5)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffff00', 
                markersize=8, label='Media (5-15)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff0000', 
                markersize=8, label='Alta (> 15)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.pause(0.001)



class Q_Network(nn.Module):

    def __init__(self, n_observations, hidden = 128, weights = None):
        
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.layer4 = nn.Linear(hidden, hidden)

        self.input_norm = nn.BatchNorm1d(n_observations)
        self.norm = nn.BatchNorm1d(hidden)

        self.criterion = nn.SmoothL1Loss()
        #self.optimizer = optim.AdamW(self.parameters(), lr = learning_rate, amsgrad=True)
        self.activation = F.relu

        self.weights = weights


    def instantiate_optimizer(self, learning_rate = 1e-4):
        self.optimizer = optim.SGD(self.parameters(), lr = learning_rate, momentum = 0.9, nesterov = True)

    
    def common_forward(self, x):

        x = self.input_norm(x)
        
        z = self.activation(self.layer1(x))
        
        y = self.activation(self.layer2(z))

        y = self.activation(self.layer3(y))

        z = z + self.activation(self.layer4(y))
        
        z = self.norm(z)

        return z
    


class Lex_Q_Network(Q_Network):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4, weights = None):
        
        super().__init__(n_observations, hidden, weights)

        self.layerb1 = nn.Linear(hidden, n_actions)
        self.layerb2 = nn.Linear(hidden, n_actions)
        self.layerb3 = nn.Linear(hidden, n_actions)

        self.I = torch.eye(hidden, device = device)
        self.I_o1 = torch.eye(self.layerb1.out_features, device = device)
        self.I_o2 = torch.eye(self.layerb1.out_features + self.layerb2.out_features, device = device)

        self.instantiate_optimizer(learning_rate)


    def forward(self, x):

        z = self.common_forward(x)

        ort2, ort3, prj2, prj3 = self.project(z)

        o1 = F.sigmoid(self.layerb1(z)) - 1

        o2 = self.layerb2(ort2 + prj2)

        o3 = self.layerb3(ort3 + prj3)

        return torch.stack((o1, o2, o3), dim = 2)
    
    # Assumption: W is column full rank. 
    def project(self, z): # https://math.stackexchange.com/questions/4021915/projection-orthogonal-to-two-vectors

        W1 = self.layerb1.weight.clone().detach()
        W2 = self.layerb2.weight.clone().detach()
        ort2 = torch.empty_like(z)
        ort3 = torch.empty_like(z)
        
        zz = z.clone().detach()

        #mask = torch.heaviside(zz, self.v)
        #Rk = torch.einsum('ij, jh -> ijh', mask, self.I)
        #W1k = W1.matmul(Rk)
        #W2k_ = W2.matmul(Rk)
        #W2k = torch.cat((W1k, W2k_), dim = 1)
        W2k = torch.cat((W1, W2), dim = 0)

        ort2 = self.compute_orthogonal(z, W1, self.I_o1)
        ort3 = self.compute_orthogonal(z, W2k, self.I_o2)

        self.ort2 = ort2.clone().detach()
        self.ort3 = ort3.clone().detach()

        prj2 = zz - self.ort2
        prj3 = zz - self.ort3

        return ort2, ort3, prj2, prj3


    def compute_orthogonal(self, z, W, I_o):
        WWT = torch.matmul(W, W.mT)
        P = solve_matrix_system(WWT, I_o)
        P = torch.matmul(P, W)
        P = self.I - torch.matmul(W.mT, P)
        
        return torch.matmul(z, P)
        
    
    def learn(self, predict, target):
        self.optimizer.zero_grad()
        loss_f = self.criterion(predict[:,0], target[:,0])
        loss_i1 = self.criterion(predict[:,1], target[:,1])
        loss_i2 = self.criterion(predict[:, 2], target[:,2])
        
        loss_f.backward(retain_graph=True)
        loss_i1.backward(retain_graph=True)
        loss_i2.backward()

        self.optimizer.step()

        #return torch.tensor([loss_f, loss_i1, loss_i2]).clone().detach()
        

    def  __str__(self):
        return "Lex"



class Weighted_Q_Network(Q_Network):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4, weights = None):

        super().__init__(n_observations, hidden, weights)
        
        self.n_actions = n_actions
        self.reward_size = len(weights)

        self.weights = self.weights.to(device)

        self.final_layer = nn.Linear(hidden, n_actions*self.reward_size)

        self.instantiate_optimizer(learning_rate)

    
    def forward(self, x):
        
        z = self.common_forward(x)

        z = self.final_layer(z).view(-1, self.n_actions, self.reward_size)

        return z


    def learn(self, predict, target):

        self.optimizer.zero_grad()

        loss = self.criterion(torch.matmul(predict, self.weights), torch.matmul(target, self.weights))

        loss.backward()

        self.optimizer.step()


    def  __str__(self):
        return "Weight"



class Scalar_Q_Network(nn.Module):

    def __init__(self, n_observations, n_actions, hidden = 128, learning_rate = 1e-4):
        pass


    def  __str__(self):
        return "Scalar"



class QAgent():

    def __init__(self, network, env, learning_rate, batch_size, hidden, slack, \
                 epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, \
                 train_start, replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, env_type):

        self.env = env

        self.batch_size = batch_size
        self.state_size = 8 #env.state_size
        self.action_size = env.action_size
        self.reward_size = env.reward_size
        self.slack = slack

        #modifiche per env testing
        self.env.env_type = env_type
        if env_type == 4:
            self.env.iter = True
            self.env.env_type = 1 # start iteration
        else:
            self.env.env_type = env_type

        self.permissible_actions = torch.tensor(range(self.action_size)).to(device)

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        self.replay_frequency = replay_frequency
        self.target_model_update_rate = target_model_update_rate
        self.mini_batches = mini_batches

        #modifiche per curriculum
        self.curriculum_stage = 1


        self.score = []
        self.epsilon_record = []
        self.completed = []
        self.num_actions = []

        self.train_start = train_start

        self.memory = deque(maxlen=memory_length)

        self.model = network(self.state_size, self.action_size, hidden, learning_rate, weights)
        self.target_model = network(self.state_size, self.action_size, hidden, learning_rate, weights)
        self.model.to(device)
        self.target_model.to(device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        self.target_model.eval()

        self.state_batch = torch.empty((self.batch_size, self.state_size), device = device)
        self.next_state_batch = torch.empty((self.batch_size, self.state_size), device = device)
        self.action_batch = torch.empty((self.batch_size), dtype = torch.long, device = device)
        self.reward_batch = torch.empty((self.batch_size,self.reward_size), device = device)
        self.not_done_batch = torch.empty(self.batch_size, dtype = torch.bool, device = device)
        self.next_state_values = torch.zeros((self.batch_size, self.reward_size), device = device)


    def update_epsilon(self):
        if len(self.memory) >= self.train_start and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
            
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.random_action()   # This could be substituted by arglexmax ahead in time

        with torch.no_grad():
            Q = self.model(state).squeeze()
        
            return self.greedy_arglexmax(Q)


    def greedy_arglexmax(self, Q):
        permissible_actions = self.permissible_actions

        mask = (Q[:, 0] >= -0.7) # sceglie solo le azioni con Q[0] >= -0.7, cioè quelle con reward collisione accettabile

        if not torch.any(mask):
            permissible_actions = self.refine_actions(permissible_actions, Q[:,0])
        
        else:
            permissible_actions = permissible_actions[mask]

        permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])
        
        return permissible_actions[Q[permissible_actions,2].max(0)[1]]


    def select_action(self, state):
        if random.random() <= self.epsilon: #<= ?????????????????????????? 
            return self.env.random_action()
        
        with torch.no_grad():
            q_value = self.model(state).squeeze()
            action = self.arglexmax(q_value)
            return action


    def refine_actions(self, permissible_actions, q):
        lower_bound = q.max(0)[0]
        lower_bound -= self.slack * torch.abs(lower_bound)
        mask = q >= lower_bound

        return permissible_actions[mask]

 
    def arglexmax(self, Q):
        permissible_actions = self.permissible_actions

        mask = (Q[:, 0] >= -0.7)

        if not torch.any(mask):
            permissible_actions = self.refine_actions(permissible_actions, Q[:,0])
        
        else:
            permissible_actions = permissible_actions[mask]

        #mask = (Q[permissible_actions, 1] > 0)

        #if not torch.any(mask):
        #    permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])
        
        #else:
        #    permissible_actions = permissible_actions[mask]
        #    permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])

        permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])

        permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,2])

        """
        for i in range(self.reward_size):
            #if len(permissible_actions) == 1:
            #    break

            lower_bound = Q[permissible_actions, i].max(0)[0]
            lower_bound -= self.slack * torch.abs(lower_bound)
            permissible_actions = [a for a in permissible_actions if Q[a, i] >= lower_bound]
        """

        return random.choice(permissible_actions)

    
    def update_target_model(self, tau):
        weights = self.model.state_dict()
        target_weights = self.target_model.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        self.target_model.load_state_dict(target_weights)
    

    def save_arglexmax(self, i):
        self.actions[i] = self.arglexmax(self.q_values[i,:])


    def q_value_arrival_state(self, states):
        self.q_values = self.model(states)
        self.actions = torch.empty(len(states), device=device, dtype=torch.int64)

        for i in range(len(states)):
            #self.actions[i] = self.arglexmax(self.q_values[i,:])
            self.actions[i] = self.greedy_arglexmax(self.q_values[i,:])
        #p.map(self.save_arglexmax, np.arange(0, len(states)))

        actions = torch.vstack((self.actions, self.actions, self.actions)).T.unsqueeze(1)
        # evaluate a' according wih target network
        return self.target_model(states).gather(1,actions).squeeze()

    
    def experience_replay(self):
        if len(self.memory) < self.train_start:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for i in np.arange(self.batch_size):
            self.state_batch[i,:] = batch[i][0]
            self.action_batch[i] = batch[i][1]
            self.reward_batch[i,:] = batch[i][2]
            self.next_state_batch[i,:] = batch[i][3]
            self.not_done_batch[i] = not batch[i][4]
            
        
        
        with torch.no_grad():
            self.next_state_values[self.not_done_batch, :] = self.q_value_arrival_state(self.next_state_batch[self.not_done_batch,:])
            target_values = self.reward_batch + self.gamma * self.next_state_values

        self.model.train()
        
        action_batch = torch.vstack((self.action_batch, self.action_batch, self.action_batch)).T.unsqueeze(1)
        predicted_values = self.model(self.state_batch).gather(1,action_batch).squeeze()

        self.model.learn(predicted_values, target_values)
        
        self.model.eval()


    def learn(self):
        bar = qqdm(np.arange(self.episodes), desc="Learning")

        best_completed = 0.0 # Track the best completition score
        consecutive_successes = 0 # counter for consecutive completed episodes

        for e in bar:
        
            state = self.env.reset()
            state = torch.tensor(state).to(device)
            episode_score = np.zeros(self.reward_size)
            step = 0
            done = False
                        
            while not done:
                #action = self.select_action(state.unsqueeze(0))
                action = self.act(state.unsqueeze(0))
                next_state, reward, terminated, truncated, completed = self.env.step(action)

                self.env.render()
                done = terminated or truncated          
                next_state = torch.tensor(next_state).to(device)
                episode_score += reward
                reward = torch.tensor(reward)
                self.add_experience(state, action, reward, next_state, terminated)
                state = next_state
                
                if (step & self.replay_frequency) == 0:
                    for i in np.arange(self.mini_batches):
                        self.experience_replay()
                        self.update_target_model(self.target_model_update_rate)
                
                step += 1

                if done:                                
                    self.update_epsilon()
                    # Passa alla fase 2 quando epsilon ≈ epsilon_min
                    if self.epsilon <= 0.02 and self.curriculum_stage == 1: #0.01 HARDCODED
                        self.curriculum_stage = 2
                        self.env.curriculum_stage = 2

                    self.score.append(episode_score)
                    self.epsilon_record.append(self.epsilon)
                    self.completed.append(completed)
                    self.num_actions.append(step)

                # Update best completion score and check conditions
                if completed:
                    # Calculate moving averages over last 31 episodes (or all available if fewer)
                    window_size = min(31, len(self.completed))
                    
                    # Collision score (index 0 in self.score)
                    current_collisions = np.mean([s[0] for s in self.score[-window_size:]]) if self.score else 0
                    
                    # Completion rate
                    current_completed = np.mean(self.completed[-window_size:]) if self.completed else 0
                    
                    # Update best completion score
                    if current_completed > best_completed:
                        best_completed = current_completed
                        print(f"New best completion score: {best_completed:.2f} at episode {e}")
                    
                    # Check for model saving condition
                    if current_collisions == 0 and current_completed > 0.70:
                        save_path = f"best_model_episode_{e}.pt"
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Model saved at episode {e}: Collision=0, Completed={current_completed:.2f}")
                    
                    # Update consecutive successes counter
                    if current_completed > 0.70:
                        consecutive_successes += 1
                    else:
                        if consecutive_successes > 0:
                            print(f"{consecutive_successes} consecutive successes reset at episode {e}.")
                            consecutive_successes = 0
                    
                    # Early stopping condition
                    if consecutive_successes >= 100:
                        print(f"Early stopping achieved at episode {e} with {consecutive_successes} consecutive successes.")
                        break
                if e >= 31:
                    rew_mean = sum(self.score[-31:]) / 31
                    compl_mean = np.mean(self.completed[-31:])
                    act_mean = np.mean(self.num_actions[-31:])
                    current_eps = self.epsilon_record[-1] if self.epsilon_record else self.epsilon

                    bar.set_infos({
                            'Speed_': f'{(time.time() - bar.start_time) / (bar.n + 1):.2f}s/it',
                            'Collision': f'{rew_mean[0]:.2f}',
                            'Forward': f'{rew_mean[1]:.2f}',
                            'OOB': f'{rew_mean[2]:.2f}',
                            'Completed': f'{compl_mean:.2f}',
                            'Actions': f'{act_mean:.2f}',
                            'Epsilon': f'{current_eps:.4f}'
                        })
                self.env.completed_mean = compl_mean

                    
        #self.env.close()


    def plot_score(self, score, start, end, N, title, filename):
        plt.plot(score);
        mean_score = np.convolve(array(score), np.ones(N)/N, mode='valid')
        plt.plot(np.arange(start,end), mean_score)

        if title is not None:
            plt.title(title);

        plt.savefig(filename)
        plt.clf()


    def plot_learning(self, N, title, filename):
        vs = VectorScore(*zip(*self.score))
        time = len(vs.oob)
        start = math.floor(N/2)
        end = time-start
        self.plot_score(vs.collision, start, end, N, title + " collision", filename + str(self.env) + "_collision_graph")
        self.plot_score(vs.oob, start, end, N, title + " oob", filename + str(self.env) + "_oob_graph")
        self.plot_score(vs.distance, start, end, N, title + " distance", filename + str(self.env) + "_distance_graph")
        self.plot_score(self.completed, start, end, N, title + " completed", filename + str(self.env) + "_completed_graph")
        self.plot_score(self.num_actions, start, end, N, title + " actions", filename + str(self.env) + "_actions_graph")
        

    def plot_epsilon(self, filename = ""):
        plt.plot(self.epsilon_record);
        plt.title("Epsilon decay");
        plt.savefig(filename + str(self.env) + "_epsilon");
        plt.clf()
        
    
    def save_model(self, path=""):
        torch.save(self.model.state_dict(), path+str(self.env)+"_"+str(self)+".pt")

    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path+str(self.env)+"_"+str(self)+".pt", map_location=torch.device(device)))
        self.model.eval()


    def __str__(self):
        return "QAgent"


    def simulate(self, number = 0, path = "", verbose = False):
        done = False
        self.epsilon = -1
        state = self.env.reset()
        state = torch.tensor(state).to(device)
        position_x = []
        position_y = []
        position_x.append(self.env.car.position[0])
        position_y.append(self.env.car.position[1])

        #print(state)
        #print(self.env.car.position)
        #print("")

        while not done:
            if verbose:
                print(state)
                print(self.model(state.unsqueeze(0)))

            #action = self.select_action(state.unsqueeze(0))
            action = self.act(state.unsqueeze(0))

            if verbose:
                print(action)
                print("")
            
            next_state, _, terminated, truncated, _  = self.env.step(action)

            done = terminated or truncated

            state = torch.tensor(next_state).to(device)
            #print(state)
            #print(self.env.car.position)
            #print("")

            position_x.append(self.env.car.position[0])
            position_y.append(self.env.car.position[1])

        plt.plot(position_x, position_y);

        # lanes
        plt.plot([0.0, self.env.dim_x], [0.0, 0.0]);
        plt.plot([0.0, self.env.dim_x], [self.env.dim_y, self.env.dim_y]);

        # jaywalker
        jaywalker_position_x = [self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] + self.env.jaywalker_r,\
                                self.env.jaywalker[0] - self.env.jaywalker_r]
        
        jaywalker_position_y = [self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] + self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r,\
                                self.env.jaywalker[1] - self.env.jaywalker_r]
        
        plt.plot(jaywalker_position_x, jaywalker_position_y);

        plt.savefig(path + str(self.env) + "_simulation_" + str(number) + "_render.png");
        plt.clf();


    def test_model(self, model_path, num_episodes=10, render=True):
        """
        Test the trained model after training.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        success_rate = 0
        collision_rate = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            state = torch.tensor(state).to(device)
            done = False

            while not done:
                if render:
                    self.env.render()

                with torch.no_grad():
                    Q = self.model(state.unsqueeze(0)).squeeze()
                    action = self.greedy_arglexmax(Q)

                next_state, _, terminated, truncated, completed = self.env.step(action)
                done = terminated or truncated
                state = torch.tensor(next_state).to(device)

            # Update success and collision rates
            success_rate += int(completed)
            collision_rate += int(terminated and not completed)

        print(f"Test Results ({num_episodes} episodes):")
        print(f"- Success Rate: {success_rate / num_episodes * 100:.2f}%")
        print(f"- Collision Rate: {collision_rate / num_episodes * 100:.2f}%")





# def main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
#                 replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations, version = ""):

#     agent = QAgent(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
#                 replay_frequency, target_model_update_rate, memory_length, mini_batches, weights)
    

#     agent.learn()
#     agent.plot_learning(31, title = "Jaywalker", filename = img_filename + str(agent.model) + "_" + version)
#     agent.plot_epsilon(img_filename + str(agent.model) + "_" + version)
    
#     for i in np.arange(num_simulations):
#         agent.simulate(i, simulations_filename + str(agent.model) + "_" + version)
    
#     agent.save_model(str(agent.model) + "_" + version)

# def main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
#                 replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations, version = ""):

#     agent = QAgent(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
#                 replay_frequency, target_model_update_rate, memory_length, mini_batches, weights)
#     agent.load_model("Lex_jaywalker_QAgent.pt")




if __name__ == "__main__":
    
    env = Jaywalker()
    episodes = 3000
    replay_frequency = 3
    gamma = 0.95
    learning_rate = 1e-2 #5e-4
    epsilon_start = 0
    epsilon_decay = 0.997 #0.997 0.995
    epsilon_min = 0.01
    batch_size = 256
    train_start = 1000
    target_model_update_rate = 1e-3
    memory_length = 10000 #100000
    mini_batches = 4
    branch_size = 256
    slack = 0.1
    hidden = 128
    num_simulations = 1
    img_filename = "imgs/"
    simulations_filename = "imgs/simulations/"
    simulations = 0
    network = Lex_Q_Network
    weights = None


    env_type = sys.argv[1]

    #p = Pool(32)
    if env_type == "1": #easy enviroment: jaywalker still and car far away 
        env_type = 1

    elif env_type == "2": #hard enviroment: jaywalker still and car close
        env_type = 2

    elif env_type == "3": #very hard enviroment: jaywalker moving and car far away
        env_type = 3
    
    elif env_type == "4": #showing every enviroment sequentially
        env_type = 4
    else:
        raise ValueError("enviroment type " + env_type + " unknown:\n" + "1 -> jaywalker still and car far away \n2 -> jaywalker still and car close\n3 -> jaywalker moving and car far away\n4 -> jaywalker moving and car close")
    
    agent = QAgent(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                   replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, env_type)
    
    agent.test_model(
        model_path="agent.pt",
        num_episodes=4,
        render=True
    )
