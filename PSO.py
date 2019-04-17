import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def sphere(pos):
    return np.power(pos, 2).sum(axis=1).reshape(pos.shape[0], 1)

class PSO():
    def __init__(self, swarm_size=50, n_dim=3, n_iter=1000, 
                obj_func=sphere, minf=0, maxf=1, c1=2.05, c2=2.05,
                 min_init=0, max_init=1, up_w=0.9, lb_w=0.4, vmax=100):

        self.obj_func = obj_func
        self.pos = np.random.uniform(min_init, max_init, (swarm_size, n_dim))
        self.vels = np.zeros((swarm_size, n_dim))

        self.pbest_cost = self.obj_func(self.pos)
        self.pbest_pos = self.pos

        self.gbest_cost = self.pbest_cost.min()
        self.gbest_pos = self.pbest_pos[self.pbest_cost.argmin()]

        self.gbest_track_iter = [self.gbest_cost]
        self.n_iter = n_iter
        self.c1 = c1
        self.c2 = c2
        self.minf = minf
        self.maxf = maxf

        self.w = up_w
        self.up_w = up_w
        self.lb_w = lb_w
        self.vmax = vmax

        # self.phi = self.c1 + self.c2 #Utilizado no caso do fator de contricao de Clerc
        # self.clerc = 2 / np.abs(2 - self.phi - np.sqrt((self.phi ** 2) - 4 * self.phi))


    def fit(self):
        for it in tqdm(range(self.n_iter)):
            r1 = np.random.random(self.pos.shape[1])
            r2 = np.random.random(self.pos.shape[1])
            
            self.vels = self.w*self.vels + r1*self.c1*(self.pbest_pos - self.pos) + r2*self.c2*(self.gbest_pos - self.pos)
            
            self.pos = self.pos + self.vels
            
            self.vels[self.vels > self.vmax] = self.vmax

            # self.vels[self.pos < self.minf] = -1 * self.vels[self.pos < self.minf]
            # self.vels[self.pos > self.maxf] = -1 * self.vels[self.pos > self.maxf]
            # self.pos[self.pos > self.maxf] = self.maxf
            # self.pos[self.pos < self.minf] = self.minf

            aux_pbest_cost = np.minimum(self.pbest_cost, self.obj_func(self.pos))
            
            new_pbest_pos = np.nonzero(self.pbest_cost - aux_pbest_cost)[0]
            
            self.pbest_pos[new_pbest_pos] = self.pos[new_pbest_pos]
            
            self.pbest_cost = aux_pbest_cost
            if self.pbest_cost.min() < self.gbest_cost:
                self.gbest_cost = self.pbest_cost.min()
                self.gbest_pos = self.pbest_pos[self.pbest_cost.argmin()]
            
            self.w = self.up_w - ((self.up_w - self.lb_w) * (it / self.n_iter))
            self.gbest_track_iter.append(self.gbest_cost)


if __name__ == '__main__':
    # n_sim = 30
    # results = pd.DataFrame()
    # for s in range(n_sim):
    #     pso = PSO(minf=-100, maxf=100, n_dim=30, swarm_size=30,
    #               n_iter=50000, min_init=50, max_init=100, up_w=0.9, lb_w=0.4)
    #     pso.fit()
    #     results[f'sim_{s}'] = pso.gbest_track_iter
    #     best = min(pso.gbest_track_iter)

    # results.to_csv(f'pso_sphere_global_{n_sim}_decain_w.csv', index=False)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # plt.plot(pso.gbest_track_iter, label=best)
    # plt.legend()
    # plt.show()

    # results = pd.read_csv('pso_sphere_global.csv')

    # f, ax = plt.subplots(2,1)
    
    # results.plot(legend=False, ax=ax[0])
    # ax[0].set_xlabel('Number of iterations')
    # ax[0].set_ylabel('Fitness')

    # results.tail(1).transpose().boxplot(ax=ax[1])
    # ax[1].set_xlabel('PSO Global')
    # ax[1].set_ylabel('Fitness')

    # plt.subplots_adjust(hspace = 0.7)
    # plt.show()
    pass
    
