import numpy as np
import math
import random
from PIL import Image, ImageDraw
import os

class Particle:
    def __init__(self, pos=np.array([0., 0.]), vel=np.array([0., 0.])):
        self.eos = True
        self.density = 1.
        self.bulk_modulus = 1.
        self.gamma = 7.
        self.mass = 1.
        self.volume = 1.
        self.x = pos
        self.vel = vel
        self.F = np.array([ [1., 0.],
                            [0., 1.]])
        self.Jp = 1.
        self.closest_cell = np.array([0, 0])
        self.weights = np.array([   [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]])
        self.dweights = np.array([   [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]])
    def update_weights(self, grid_res):
        dx = 1. / np.float32(grid_res)
        self.closest_cell = self.get_closest_cell(grid_res)
        X_eval = self.x - (self.get_cell_center(grid_res, self.closest_cell) - dx)
        for axis in range(2):
            one_over_dx = np.float32(grid_res)
            axis_x = X_eval[axis]
            self.compute_weight(axis_x * one_over_dx, one_over_dx, axis)
    
    def get_closest_cell(self, grid_res):
        inv_dx = np.float32(grid_res)
        dx = 1. / np.float32(grid_res)
        i, j = math.floor(self.x[0] * inv_dx), math.floor(self.x[1] * inv_dx)
        return np.array([i, j])

    def get_cell_center(self, grid_res, cell_idx):
        dx = 1. / np.float32(grid_res)
        half_dx = .5 * dx
        posx, posy = half_dx + cell_idx[0] * dx, half_dx + cell_idx[1] * dx
        return np.array([posx, posy])

    def compute_weight(self, x, one_over_dx, k):
        self.weights[0][k]=abs(0.5*x*x-1.5*x+1.125)
        self.dweights[0][k]=(x-1.5)*one_over_dx
        x-=1.
        self.weights[1][k]=abs(-x*x+.75)
        self.dweights[1][k]=(-2.)*x*one_over_dx
        x-=1.
        self.weights[2][k]=abs(.5*x*x+1.5*x+1.125)
        self.dweights[2][k]=(x+1.5)*one_over_dx

    def get_weight(self, neighbor_idx):
        cur_weight = 1.
        for i in range(2):
            cur_weight *= self.weights[neighbor_idx[i]][i]
        return cur_weight

    def get_weight_gradient(self, neighbor_idx):
        wg=np.array([1., 1.])
        for i in range(2):
            for j in range(2):
                if i==j:
                    wg[i] *= self.dweights[neighbor_idx[j]][j]
                else:
                    wg[i] *= self.weights[neighbor_idx[j]][j]
        return wg

    def precompute_material_state(self):
        pass

class Simulator:
    def __init__(self):
        self.wall_width = 0.1
        self.radius = 0.1
        self.n = 256
        self.n_cells = self.n**2
        self.npc = 4
        self.n_particles = np.int(self.npc * math.pi * self.radius**2 * self.n_cells)
        

        self.output_path = './output_%04d' % self.n
        os.system('rm -rf ' + self.output_path + '/*')
        
        if not os.path.exists(self.output_path):
            os.system('mkdir ' + self.output_path)
        
        self.dt = 1e-4
        self.frame_dt = 1e-3
        self.dx = 1.0 / self.n
        self.inv_dx = 1.0 / self.dx

        # Material properties
        self.particle_mass = 1.0
        self.vol = 1.0                       # Particle Volume
        self.hardening = 10.0               # Snow hardening factor
        self.E = 100              # Young's Modulus
        self.nu = 0.2             # Poisson ratio

        # Initial Lamé parameters
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lam_0 = self.E * self.nu / ((1+self.nu) * (1 - 2 * self.nu))
        self.mu = self.mu_0
        self.lam = self.lam_0
        self.particles = []
        self.grid_mass = np.zeros(self.n_cells).reshape(self.n, self.n)
        self.grid_vel = np.zeros(self.n_cells * 2).reshape(self.n, self.n, 2)
        self.grid_vel_star = np.zeros(self.n_cells * 2).reshape(self.n, self.n, 2)
        self.add_object(np.array([0.5, 0.25]), self.radius, np.array([0., -3.]))
        self.save()
        self.frame = 0
        self.n_steps = 10000
        

    def reset_grid_vars(self):
        self.grid_mass = np.zeros(self.n_cells).reshape(self.n, self.n)
        self.grid_vel = np.zeros(self.n_cells * 2).reshape(self.n, self.n, 2)
    
    def update_particle_weights(self):
        for pid in range(self.n_particles):
            p = self.particles[pid]
            p.update_weights(self.n)

    def p2g(self):
        for pid in range(self.n_particles):
            p = self.particles[pid]
            closest_cell = p.closest_cell
            p_mass = p.mass
            p_vel = p.vel
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    d_idx = np.array([di, dj])
                    cur_cell = closest_cell + d_idx
                    if self.oob(cur_cell):
                        print('Out of boundary in p2g')
                    else:
                        cur_weight = p.get_weight(np.array([di+1, dj+1]))
                        self.grid_mass[cur_cell[0]][cur_cell[1]] += cur_weight * p_mass
                        self.grid_vel[cur_cell[0]][cur_cell[1]] += cur_weight * p_mass * p_vel
        
        for i in range(self.n):
            for j in range(self.n):
                g_mass = self.grid_mass[i][j]
                if g_mass > 0.:
                    self.grid_vel[i][j] /= g_mass

    def update_material_state(self):
        for pid in range(self.n_particles):
            p = self.particles[pid]
            if not p.eos:
                p.precompute_material_state()

    def apply_force(self):
        gravity = np.array([0., -2.])
        grid_force = np.zeros(self.n_cells * 2).reshape(self.n, self.n, 2)
        for pid in range(self.n_particles):
            p = self.particles[pid]
            p_mass = p.mass
            p_vel = p.vel
            closest_cell=p.closest_cell
            eos_coefficient=0.
            V0=p.volume; 
            if p.eos:
                eos_coefficient=V0/p.density*p.bulk_modulus*(p.density**(p.gamma - 1.))
            else:
                pass
            
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    d_idx = np.array([di, dj])
                    cur_cell = closest_cell + d_idx
                    if self.oob(cur_cell):
                        print('Out of boundary in apply_force')
                    else:
                        cur_weight = p.get_weight(np.array([di+1, dj+1]))
                        cur_wg = p.get_weight_gradient(np.array([di+1, dj+1]))
                        body_force = p_mass * gravity * cur_weight
                        inner_force = eos_coefficient * cur_wg
                        grid_force[cur_cell[0]][cur_cell[1]] += body_force
                        grid_force[cur_cell[0]][cur_cell[1]] += inner_force
        dt = self.dt
        for i in range(self.n):
            for j in range(self.n):
                g_mass = self.grid_mass[i][j]
                if g_mass > 0.:
                    self.grid_vel_star[i][j] = self.grid_vel[i][j] + dt * grid_force[i][j] / g_mass
        self.grid_based_collision()

    def grid_based_collision(self):
        dx = 1. / self.n
        half_dx = .5 * dx
        for i in range(self.n):
            for j in range(self.n):
                cur_cell = np.array([i, j])
                posx, posy = half_dx + i * dx, half_dx + j * dx
                if self.collided(posx, posy):
                    self.grid_vel_star[i][j] = np.array([0., 0.])

    def outer_product(self, col, row):
        mat = np.array([[0., 0.], [0., 0.]])
        for i in range(2):
            for j in range(2):
                mat[i][j] = col[i] * row[j]
        return mat

    def trace(self, mat):
        return mat[0][0] + mat[1][1]

    def collided(self, posx, posy):
        ww = self.wall_width
        return posx < ww or posx > 1. - ww or posy < ww or posy > 1. - ww

    def g2p(self):
        flip = 0.95
        self.apply_force()
        for pid in range(self.n_particles):
            p = self.particles[pid]
            closest_cell = p.closest_cell
            V_pic = np.array([0., 0.])
            V_flip = p.vel
            grad_Vp = np.array([[0., 0.], [0., 0.]])
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    d_idx = np.array([di, dj])
                    cur_cell = closest_cell + d_idx
                    if self.oob(cur_cell):
                        print('Out of boundary in g2p')
                    else:
                        cur_weight = p.get_weight(np.array([di+1, dj+1]))
                        cur_wg = p.get_weight_gradient(np.array([di+1, dj+1]))
                        V_grid = self.grid_vel_star[cur_cell[0]][cur_cell[1]]
                        delta_V_grid = self.grid_vel_star[cur_cell[0]][cur_cell[1]] - self.grid_vel[cur_cell[0]][cur_cell[1]]
                        V_pic += cur_weight * V_grid
                        V_flip += cur_weight * delta_V_grid
                        grad_Vp += self.outer_product(V_grid, cur_wg)
            if p.eos:
                p.density /= 1. + self.dt * self.trace(grad_Vp)
            else:
                pass
            p.vel = V_flip * flip + V_pic * (1.-flip)
            p.x += V_pic * self.dt

    def advance_step(self):
        self.reset_grid_vars()
        self.update_particle_weights()
        self.p2g()
        self.update_material_state()
        self.g2p()
    
    def run(self):
        for step in range(self.n_steps):
            self.advance_step()
            if step % 100 == 0:
                self.save(self.frame)
                self.frame += 1

    def oob(self, cell_idx):
        return cell_idx[0]<0 or cell_idx[0]>=self.n or cell_idx[1]<0 or cell_idx[1]>=self.n
    
    def add_object(self, center, radius, vel):
        n_particles = self.n_particles
        for i in range(n_particles):
            posx, posy = (2.*random.random()-1.) * radius , (2.*random.random()-1.) * radius
            dis = math.sqrt(posx**2 + posy**2)
            while dis > radius:
                posx, posy = (2.*random.random()-1.) * radius , (2.*random.random()-1.) * radius
                dis = math.sqrt(posx**2 + posy**2)
            p = Particle(np.array([posx, posy]) + center, vel)
            self.particles.append(p)

    def save(self, step=-1):
        print('saving frame {}'.format(step))
        n_particles = self.n_particles
        w = 2048
        h = 2048
        # print('# particles: {}'.format(n_particles))
        im = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        e_radius = 20
        for i in range(n_particles):
            p = self.particles[i]
            pos = p.x
            draw.ellipse((w * pos[0], h - h * pos[1], w * pos[0] + e_radius, h - h * pos[1] + e_radius), fill=(255, 255, 255), outline=(255, 255, 255))
	    	# draw.ellipse((w * pos[0], h - h * pos[1], w * pos[0] + 1, h - h * pos[1] + 1), fill=(255, 255, 255), outline=(255, 255, 255))
        if step == -1:
            im.save(self.output_path + '/init.bmp', quality=95)
        else: 
            im.save(self.output_path + '/{:04d}'.format(step)+'.bmp', quality=95)


simulator = Simulator()
simulator.run()