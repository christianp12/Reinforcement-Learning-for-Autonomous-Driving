import numpy as np
from qqdm import qqdm
import matplotlib.pyplot as plt
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
if torch.cuda.is_available():
    device = "cuda:0"
else:
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

# namedtuple creates an immutable class similar to a tuple but with named fields
# collision: indicates whether a collision occurred
# distance: indicates a numerical value indicating a distance measurement
# oob: "out of bounds" indicates if the car is outside of the lane
VectorScore = namedtuple('VectorScore', ('collision', 'distance', 'oob'))


# partiamo con azioni discrete
# reward = [collision, distance from target, centered in own lane]
# speed is saturated to 50

def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


class Car:
    def __init__(self, position):

        self.lf = 1 # front axle
        self.lr = 1 # rear axle

        self.max_speed = 20.0

        self.reset(position)

    # a: acceleration
    # df: delta front (steering angle)
    def move(self, df, a):

        self.v += a
        
        # speed saturation: keeps valocity in realistic range
        self.v = np.maximum(-self.max_speed, np.minimum(self.max_speed, self.v)) 
        
        # update slip angle according to steering input
        #self.beta = np.arctan((1 + self.lr / self.lf) * np.tan(df))
        self.beta += np.arctan((1 + self.lr / self.lf) * np.tan(df))

        
        arg = self.phi + self.beta # car's heading angle + slip angle

        self.prev_position = np.copy(self.position) # store previous position
        self.position += self.v * array([np.cos(arg), np.sin(arg)]) # updates position (move car)

        #self.phi += self.v/self.lr * np.sin(self.beta)     ## Stearing inertia disabled
        #ang = array([np.cos(self.phi), np.sin(self.phi)])

        ang = array([np.cos(self.beta), np.sin(self.beta)]) # allign axle offsets to car's motion direction

        self.prev_front = self.front
        self.front = self.position + self.lf * ang # update front axle

        #self.prev_back = self.back
        self.back = self.position - self.lr * ang # update rear axle


    def reset(self, position):
        self.position = position

        self.v = 0

        self.phi = 0
        self.beta = 0

        self.front = array([self.position[0] + self.lf, self.position[1]])
        self.back = array([self.position[0], self.position[1] + self.lr])

        self.prev_front = np.copy(self.front)
        #self.prev_back = np.copy(self.back)



class Jaywalker:

    def __init__(self):

        self.reward_size = 3

        # road length and width
        self.dim_x = 100
        self.dim_y = 10

        # target position for the car
        self.goal = array([self.dim_x, self.dim_y/4])

        # jaywalker position
        self.jaywalker = array([self.dim_x/2, self.dim_y/4])
        # jaywalker radius (collision boundary around the pedestrian)
        self.jaywalker_r = 2

        # collision box
        self.jaywalker_max = self.jaywalker + self.jaywalker_r
        self.jaywalker_min = self.jaywalker - self.jaywalker_r

        # CAR CONTROL
        #self.df = [-np.pi/9, -np.pi/18, 0, np.pi/18, np.pi/9]
        self.df = [-np.pi/18, 0, np.pi/18]  # steering angle (small left, straight, small right)
        self.a = [-5, 0, 5] # acceleration (deceleration, no acceleration, acceleration)
        self.actions = [p for p in itertools.product(self.df, self.a)] # matrix of all possible actions

        self.num_df_actions = len(self.df)
        self.num_a_actions = len(self.a)

        self.state_size = 5 # [car position, distance from jaywalker, angle w.r.t. jaywalker, speed, steering angle]
        self.action_size = self.num_df_actions * self.num_a_actions # number of actions

        self.max_iterations = 100 # episode terminates after this number of steps

        self.car = Car(array([0.0,2.5])) # car starting position (x,y)

        self.counter_iterations = 0

        #self.prev_center_disance = np.abs(self.car.position[1] - self.goal[1])
        self.prev_target_distance = np.linalg.norm(self.car.front - self.goal) # euclidean distance from target

        self.noise = 1e-5 # small random perdurbation for exploration
        self.sight = 40 # sight range: how far the car can see in front of it

        self.scale_factor = 100


    # CHECK FOR COLLISION:
    def collision_with_jaywalker(self): 

        front = np.maximum(self.car.front, self.car.prev_front)
        prev = np.minimum(self.car.front, self.car.prev_front)

        denom = front - prev + self.noise

        # projects the jaywalker bounding box on the car's motion direction
        upper = (self.jaywalker_max - prev) / denom
        lower = (self.jaywalker_min - prev) / denom

        scalar_upper = np.min(upper)
        scalar_lower = np.max(lower)
        
        # check if jaywalker bounding box overlaps with the car's motion direction [0,1]
        if scalar_upper >= 0 and scalar_lower <= 1 and scalar_lower <= scalar_upper:
                return True
        
        return False


    # return the inverse of the distance from the jaywalker and the angle w.r.t to it
    def vision(self):

        vector_to_jaywalker = self.jaywalker - self.car.position + self.noise
        distance = np.linalg.norm(vector_to_jaywalker)

        if self.car.position[0] >= self.jaywalker[0] or distance > self.sight: ## Careful: it may still hit it
            return 0, -np.pi

        angle = np.arctan(vector_to_jaywalker[1]/vector_to_jaywalker[0])

        inv_distance = 1/distance

        return inv_distance, angle 


    def step(self, action):

        df, a = self.actions[action]

        self.car.move(df, a)

        reward = np.zeros(self.reward_size)
        terminated = False
        completed = False

        # collision with jaywalker
        if self.collision_with_jaywalker(): 
            reward[0] = -10
            terminated = True


        # distance from target
        # rewards as long as the car is moving towards the target
        reward[1] = (self.car.position[0] - self.car.prev_position[0])/self.scale_factor

        # accept surpassing the goal, terminate
        if self.car.front[0] >= self.goal[0]:
            if not terminated:
                completed = True
            terminated = True


        # collision with borders of the road -> out of bounds
        if self.car.front[1] > self.dim_y or self.car.front[1] < 0 or self.car.back[1] > self.dim_y or self.car.back[1] < 0 or self.car.position[0] < 0 or self.car.front[0] < 0:
            reward[2] = -1000
            terminated = True

        # distance from center of own lane
        else:
            # computes a distance-based penalty to encourage the car to stay centered in its lane
            reward[2] = -np.abs(self.car.position[1] - self.goal[1])

        reward[2] /= self.scale_factor * 10


        inv_distance, angle = self.vision()

        state = array([self.car.position[1], inv_distance, angle, self.car.v, self.car.beta]) # self.car.phi])

        self.counter_iterations += 1
        truncated = False

        if self.counter_iterations >= self.max_iterations:
            truncated = True


        return state, reward, terminated, truncated, completed


    def reset(self):
        self.car.reset(array([0.0,2.5]))
        self.counter_iterations = 0

        inv_distance, angle = self.vision()

        return array([self.car.position[1], inv_distance, angle, self.car.v, self.car.phi])
    

    def random_action(self):
        return int(np.floor(random.random() * self.action_size))


    def __str__(self):
        return "jaywalker"



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
                 train_start, replay_frequency, target_model_update_rate, memory_length, mini_batches, weights):

        self.env = env

        self.batch_size = batch_size
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.reward_size = env.reward_size
        self.slack = slack

        self.permissible_actions = torch.tensor(range(self.action_size)).to(device)

        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.gamma = gamma
        self.replay_frequency = replay_frequency
        self.target_model_update_rate = target_model_update_rate
        self.mini_batches = mini_batches

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

        mask = (Q[:, 0] >= -0.7)

        if not torch.any(mask):
            permissible_actions = self.refine_actions(permissible_actions, Q[:,0])
        
        else:
            permissible_actions = permissible_actions[mask]

        permissible_actions = self.refine_actions(permissible_actions, Q[permissible_actions,1])
        
        return permissible_actions[Q[permissible_actions,2].max(0)[1]]


    def select_action(self, state):
        if random.random() <= self.epsilon:
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
                    self.score.append(episode_score)
                    self.epsilon_record.append(self.epsilon)
                    self.completed.append(completed)
                    self.num_actions.append(step)

            if e >= 31:
                rew_mean = sum(self.score[-31:])/31
                compl_mean = np.mean(self.completed[-31:])
                act_mean = np.mean(self.num_actions[-31:])
                bar.set_infos({'Speed_': f'{(time.time() - bar.start_time) / (bar.n+1):.2f}s/it',
                                 'Collision': f'{rew_mean[0]:.2f}', 'Forward': f'{rew_mean[1]:.2f}', 'OOB': f'{rew_mean[2]:.2f}',
                                        'Completed': f'{compl_mean:.2f}', "Actions": f'{act_mean:.2f}'})
                    
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
        self.plot_score(vs.collision, start, end, N, title + " collision", filename + str(self.env) + "_collision")
        self.plot_score(vs.oob, start, end, N, title + " oob", filename + str(self.env) + "_oob")
        self.plot_score(vs.distance, start, end, N, title + " distance", filename + str(self.env) + "_distance")
        self.plot_score(self.completed, start, end, N, title + " completed", filename + str(self.env) + "_completed")
        self.plot_score(self.num_actions, start, end, N, title + " actions", filename + str(self.env) + "_actions")
        

    def plot_epsilon(self, filename = ""):
        plt.plot(self.epsilon_record);
        plt.title("Epsilon decay");
        plt.savefig(filename + str(self.env) + "_epsilon");
        plt.clf();
        
    
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

        plt.savefig(path + str(self.env) + "_simulation_" + str(number));
        plt.clf();


def main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations, version = ""):

    agent = QAgent(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights)
    

    agent.learn()
    agent.plot_learning(31, title = "Jaywalker", filename = img_filename + str(agent.model) + "_" + version)
    agent.plot_epsilon(img_filename + str(agent.model) + "_" + version)
    
    for i in np.arange(num_simulations):
        agent.simulate(i, simulations_filename + str(agent.model) + "_" + version)
    
    agent.save_model(str(agent.model) + "_" + version)


if __name__ == "__main__":
    
    env = Jaywalker()
    episodes = 3000
    replay_frequency = 3
    gamma = 0.95
    learning_rate = 1e-2 #5e-4
    epsilon_start = 1
    epsilon_decay = 0.997 #0.995
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

    network_type = sys.argv[1]

    #p = Pool(32)
    if network_type == "lex":
        network = Lex_Q_Network
        weights = None

    elif network_type == "weighted":
        network = Weighted_Q_Network
        weights = torch.tensor([1.0, 0.1, 0.01])

        simulations = int(sys.argv[2])

        if simulations > 1:
            weights_list = [weights ** i for i in np.arange(1, simulations+1)]
            img_filename = "weighted_simulations/" + img_filename
            simulations_filename = "weighted_simulations/simulations/"

    elif network_type == "sclar":
        network = Scalar_Q_Network

    else:
        raise ValueError("Network type" + network_type + "unknown")

    if simulations > 1:
        for i in np.arange(simulations):
            w = weights_list[i]

            main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, w, img_filename, simulations_filename, num_simulations, "v" + str(i) + "_")
    else:
        main_body(network, env, learning_rate, batch_size, hidden, slack, epsilon_start, epsilon_decay, epsilon_min, episodes, gamma, train_start,
                replay_frequency, target_model_update_rate, memory_length, mini_batches, weights, img_filename, simulations_filename, num_simulations)
