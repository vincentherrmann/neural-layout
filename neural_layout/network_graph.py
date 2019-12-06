import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import threading
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("failed importing pyplot!")
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import time
import copy
import math
from neural_layout.barnes_hut_tree import *


class Network(object):
    def __init__(self, num_dim=2):
        super().__init__()
        self.num_dim = num_dim
        self.layers = {}
        self.connections = None
        self.connection_weights = None
        self.layer_connections = {}
        self._num_units = 0
        self.parent_graph = None
        self.expand_lookup = None
        self.positions = None
        self.weights = None
        self.colors = None
        self.flat_connections = None

    @property
    def num_units(self):
        return self._num_units

    @property
    def num_connections(self):
        try:
            return self.connections.shape[0]
        except:
            return 0

    def add_layer(self, name, shape, positions=None, weights=None, colors=None):
        try:
            len(shape)
        except:
            shape = [shape]
        layer_units = np.prod(shape)
        indices = torch.arange(self.num_units, self.num_units + layer_units, dtype=torch.long)
        indices = indices.view(shape)
        self._num_units += layer_units
        self.layers[name] = indices
        self.layer_connections[name] = []

        if positions is None:
            positions = torch.randn([layer_units, self.num_dim])
        if weights is None:
            weights = torch.ones(layer_units)
        if colors is None:
            colors = torch.ones([layer_units, 4])

        if self.positions is None:
            self.positions = positions
        else:
            self.positions = torch.cat([self.positions, positions], dim=0)

        if self.weights is None:
            self.weights = weights
        else:
            self.weights = torch.cat([self.weights, weights], dim=0)

        if self.colors is None:
            self.colors = colors
        else:
            self.colors = torch.cat([self.colors, colors], dim=0)

    def add_connections(self, out_indices, connections, weights):
        if weights is None:
            weights = torch.ones_like(connections, dtype=torch.float)
        else:
            if weights.shape != connections.shape:
                raise Exception("Wrong connection weight shape")

        flat_connections = torch.stack([out_indices.view(-1, 1).repeat([1, connections.shape[1]]), connections], dim=2)
        flat_connections = flat_connections.view(-1, 2)
        allow = flat_connections[:, 1] >= 0
        flat_connections = flat_connections[allow]
        weights = weights.flatten()[allow]

        if self.connections is None:
            self.connections = flat_connections
        else:
            self.connections = torch.cat([self.connections, flat_connections], dim=0)

        if self.connection_weights is None:
            self.connection_weights = weights.flatten()
        else:
            self.connection_weights = torch.cat([self.connection_weights, weights], dim=0)

    def add_full_connections(self, input_layer, output_layer, weights=None):
        in_indices = self.layers[input_layer].flatten()
        out_indices = self.layers[output_layer].flatten()
        connections = in_indices.repeat([len(out_indices), 1])

        self.layer_connections[input_layer].append(output_layer)
        self.add_connections(out_indices, connections, weights)

    def add_one_to_one_connections(self, input_layer, output_layer, weights=None):
        out_indices = self.layers[output_layer].flatten()
        connections = out_indices[:, None] #in_indices.repeat([len(out_indices), 1])

        self.layer_connections[input_layer].append(output_layer)
        self.add_connections(out_indices, connections, weights)

    def add_conv1d_connections(self, input_layer, output_layer, kernel_size, stride=1, padding=(0, 0)):
        # input layer:  channel, time
        # output layer: channel, time

        in_indices = self.layers[input_layer]
        out_indices = self.layers[output_layer]

        in_channels, in_length = in_indices.shape
        out_channels, out_length = out_indices.shape

        # fully connected in channel dimension
        # each output channel connects to every input channel
        channel_connections = torch.arange(in_channels)[None, None, :].repeat(out_channels, out_length, kernel_size)
        # channel_connections shape: out_channels, out_length, in_channels*kernel_size

        kernel_offset = torch.arange(kernel_size)[:, None].repeat(1, in_channels).view(-1)  # kernel_size * in_channels
        time_connections = (torch.arange(out_length) * stride)[:, None] + kernel_offset[None] - padding[0]  # out_length, kernel_size * in_channels
        time_connections[time_connections < 0] = -1
        time_connections[time_connections >= in_length] = -1
        time_connections = time_connections[None].repeat(out_channels, 1, 1)  # out_channels, out_length, in_channels*kernel_size

        tc = (torch.arange(out_length) * stride)[:, None] + torch.arange(kernel_size)[None, :] - padding[0]
        tc[tc < 0] = -1
        tc[tc >= in_length] = -1
        tc = tc[None, :, :].repeat(out_channels, 1, in_channels)

        connections = in_indices[channel_connections, time_connections]
        connections[time_connections < 0] = -1
        connections = connections.view(-1, in_channels*kernel_size)
        weights = torch.ones_like(connections, dtype=torch.float)

        self.layer_connections[input_layer].append(output_layer)
        self.add_connections(out_indices, connections, weights)

    def add_conv2d_connections(self, input_layer, output_layer, kernel_size, stride=(1, 1), padding=(0, 0, 0, 0)):
        # input layer:  channel, width, height
        # output layer: channel, width, height

        in_indices = self.layers[input_layer]
        out_indices = self.layers[output_layer]

        in_channels, in_width, in_height = in_indices.shape
        out_channels, out_width, out_height = out_indices.shape

        # fully connected in channel dimension
        # each output channel connects to every input channel
        channel_connections = torch.arange(in_channels)[None, None, None, :, None].repeat(out_channels,
                                                                                    out_width,
                                                                                    out_height,
                                                                                    1,
                                                                                    kernel_size[0]*kernel_size[1])
        channel_connections = channel_connections.view(out_channels, out_width, out_height, -1)
        # channel_connections shape: out_channels, out_width, out_height, in_channels*kernel_size[0]*kernel_size[1]

        # width connections
        width_connections = (torch.arange(out_width) * stride[0])[:, None] + torch.arange(kernel_size[0])[None, :] - \
                            padding[0]  # out_width, kernel_size[0]
        width_connections[width_connections < 0] = -1
        width_connections[width_connections >= in_width] = -1
        width_connections = width_connections[None, :, None, None, :, None].repeat(out_channels, 1, out_height, in_channels, 1, kernel_size[1])  # out_channels, out_width, out_height, in_channels*kernel_size[0]*kernel_size[1]
        width_connections = width_connections.view(out_channels, out_width, out_height, -1)

        # height connections
        height_connections = (torch.arange(out_height) * stride[1])[:, None] + torch.arange(kernel_size[1])[None, :] - \
                            padding[2]  # out_height, kernel_size[1]
        height_connections[height_connections < 0] = -1
        height_connections[height_connections >= in_height] = -1
        height_connections = height_connections[None, None, :, None, None, :].repeat(out_channels, out_width, 1,
                                                                                     in_channels, kernel_size[0], 1)  # out_channels, out_width, out_height, in_channels*kernel_size[0]*kernel_size[1]
        height_connections = height_connections.view(out_channels, out_width, out_height, -1)

        connections = in_indices[channel_connections, width_connections, height_connections]
        connections[width_connections < 0] = -1
        connections[height_connections < 0] = -1
        connections = connections.view(-1, in_channels*kernel_size[0]*kernel_size[1])
        weights = torch.ones_like(connections, dtype=torch.float)

        self.layer_connections[input_layer].append(output_layer)
        self.add_connections(out_indices, connections, weights)


    def collapse_layers(self, factor=2, dimension=0):
        collapsed_graph = Network()
        collapsed_graph.parent_graph = self

        # lookup: parent node index -> child node index
        expand_lookup = torch.zeros(self.num_units, dtype=torch.long, device=self.positions.device)
        for name, indices in self.layers.items():
            # if collapsing is not possible‚
            if dimension >= len(indices.shape) or indices.shape[dimension] < factor:
                collapse_factor = 1
                print("cannot collapse dimension " + str(dimension) + " in layer " + name)
                d = 0
            else:
                collapse_factor = factor
                d = dimension

            # create reverse lookup: child node index -> parent node indices
            # reshape indices with additional collapse dimension, use zero padding if the shape does not fir
            padding = [0] * (len(indices.shape) * 2)
            target_size = math.ceil(indices.shape[d] / collapse_factor) * collapse_factor
            padding[len(padding) - 1 - d*2] = target_size - indices.shape[d]
            padded_indices = F.pad(indices, padding, value=-1)
            view_shape = list(padded_indices.shape)
            view_shape[d] = padded_indices.shape[d] // collapse_factor
            view_shape.insert(d + 1, collapse_factor)
            collapse_lookup = padded_indices.view(view_shape)
            print("layer " + name + " view_shape:" + str(view_shape))

            # move the collapse dimension to the end
            permutation = list(range(len(view_shape)))
            permutation.pop(d+1)
            permutation.append(d+1)
            collapse_lookup = collapse_lookup.permute(permutation)
            collapsed_layer_shape = list(collapse_lookup.shape)
            collapsed_layer_shape.pop(-1)
            collapse_lookup = collapse_lookup.contiguous().view(-1, collapse_factor)

            collapsed_graph.add_layer(name, shape=collapsed_layer_shape)

            new_indices = collapsed_graph.layers[name].view(-1, 1).repeat(1, collapse_factor).flatten()
            new_indices = new_indices.to(self.positions.device)
            allow = collapse_lookup.flatten() >= 0
            expand_lookup[collapse_lookup.flatten()[allow]] = new_indices[allow]
        collapsed_graph.expand_lookup = expand_lookup
        collapsed_graph.to(self.positions.device)

        collapse_num = torch.zeros_like(collapsed_graph.weights)
        collapse_num.scatter_add_(0, expand_lookup, torch.ones_like(expand_lookup, dtype=torch.float))
        collapsed_graph.positions *= 0
        collapsed_graph.positions.scatter_add_(0, expand_lookup.unsqueeze(1).repeat([1, self.num_dim]), self.positions)
        collapsed_graph.positions /= collapse_num.unsqueeze(1)
        collapsed_graph.weights *= 0
        collapsed_graph.weights.scatter_add_(0, expand_lookup, self.weights)

        nu = collapsed_graph.num_units
        print("old num units", self.num_units)
        print("old max connections index:", self.connections.max())
        new_connections = expand_lookup[self.connections]
        print("num units:", nu)
        print("max connection index:", torch.max(new_connections))

        # create a hash for the new connection (because the unique operation is very slow for two dimensions)
        connection_hash = new_connections[:, 0] * torch.LongTensor([nu]).to(new_connections.device) + new_connections[:, 1]
        unique_connections, connections_inverse = torch.unique(connection_hash, sorted=False, return_inverse=True)
        new_connections = torch.stack([unique_connections / int(nu), unique_connections % nu], dim=1)
        print("max connection index:", torch.max(new_connections))
        collapsed_graph.connections = new_connections
        new_weights = torch.zeros(unique_connections.shape[0], device=self.positions.device)
        new_weights.scatter_add_(0, connections_inverse, self.connection_weights)
        collapsed_graph.connection_weights = new_weights

        collapsed_graph.expand_lookup = expand_lookup
        return collapsed_graph

    def give_positions_to_parent(self, perturbation=1e-2):
        if self.parent_graph is None:
            print("No parent graph available")
            return None
        self.parent_graph.positions = self.positions[self.expand_lookup, :]
        self.parent_graph.positions += (torch.rand_like(self.parent_graph.positions) * 2. - 1.) * perturbation
        return self.parent_graph

    def input_connection_weight_per_unit(self):
        connection_weight = torch.zeros(self.num_units, device=self.positions.device)
        connection_weight.scatter_add_(0, self.connections[:, 1], self.connection_weights)
        return connection_weight

    def output_connection_weight_per_unit(self):
        connection_weight = torch.zeros(self.num_units, device=self.positions.device)
        connection_weight.scatter_add_(0, self.connections[:, 0], self.connection_weights)
        return connection_weight

    def to(self, device):
        self.positions = self.positions.to(device)
        self.weights = self.weights.to(device)
        if self.connections is not None:
            self.connections = self.connections.to(device)
            self.connection_weights = self.connection_weights.to(device)
        for key, value in self.layers.items():
            self.layers[key] = value.to(device)
        if self.expand_lookup is not None:
            self.expand_lookup.to(device)
        if self.parent_graph is not None:
            self.parent_graph.to(device)

    # PLOTTING #########
    def line_data(self):
        origin_positions = self.positions[self.connections[:, 0], :]
        target_positions = self.positions[self.connections[:, 1], :]
        lines = torch.stack([origin_positions, target_positions], dim=1)
        return lines

    def plot_on(self, plot, plot_connections=True):
        #fig, ax = plt.subplots(1, 1)
        try:
            plot.lines.remove()
            plot.scatter.remove()
        except:
            pass
        if plot_connections:
            plot.lines = mc.LineCollection(self.line_data(), lw=0.5, alpha=0.2)
            plot.ax.add_collection(plot.lines)
        plot.scatter = plot.ax.scatter(self.positions[:, 0], self.positions[:, 1], linewidths=1, c=self.colors)
        plot.ax.autoscale()
        plot.fig.canvas.draw()
        plt.show()

    def set_default_colors(self, colormap='Set1'):
        cmap = matplotlib.cm.get_cmap(colormap)
        layer_wise_coloring = False
        try:
            layer_wise_coloring = True
            num_colors = len(cmap.colors)
        except:
            layer_wise_coloring = False
        for l, (name, indices) in enumerate(self.layers.items()):
            if layer_wise_coloring:
                color = torch.FloatTensor(cmap(l % num_colors))
                i = indices.flatten()
                self.colors[i, :] = color
            else:
                i = indices.flatten()
                colors = torch.from_numpy(cmap(i.cpu().float() / self.num_units))
                self.colors[i, :] = colors.float()


class NetworkForceLayout:
    def __init__(self,
                 network,
                 num_dim=2,
                 repulsion=-0.005,
                 spring_optimal_distance=0.01,
                 centering=0.,
                 drag=1.,
                 noise=0.,
                 attraction_normalization=0.,
                 step_size=0.1,
                 step_discount_factor=0.9,
                 device='cpu',
                 mac=0.7,
                 force_limit=0.1,
                 distance_exponent=2.):
        self.network = network
        self.device = device
        self.mac = mac
        #self.network.to(device)
        self.num_dim = num_dim
        self.x = network.positions #torch.randn([self.network.num_units, self.num_dim], device=self.device)
        self.weights = network.weights
        self.connection_weights = network.connection_weights
        self.v = torch.zeros_like(self.x)
        self.a = torch.zeros_like(self.x)
        self.movable = torch.ones(self.network.num_units, device=self.device)
        self.per_unit_connection_weight = (self.network.input_connection_weight_per_unit()
                                           + self.network.output_connection_weight_per_unit()).to(self.device)
        self.avg_connection_count = torch.mean(self.per_unit_connection_weight)
        self.force_limit = force_limit

        self.repulsion = repulsion
        self.spring_optimal_distance = spring_optimal_distance
        self.centering = centering
        self.drag = drag
        self.noise = noise
        self.attraction_normalization = attraction_normalization
        self.step_size = step_size
        self.step_discount_factor = step_discount_factor
        self.max_levels = 12
        self.energy = float("inf")
        self.energy_progress = 0
        self.energy_progress = 0
        self.distance_exponent = distance_exponent

    def repulsion_function(self, m1, m2, difference, distance):
        return (m1 * m2 / distance ** self.distance_exponent).unsqueeze(1) * difference

    def set_position(self, indices, pos, fix=False):
        i = indices.flatten()
        self.x[i, :] = pos
        if fix:
            self.movable[i] = 0.

    def simulation_step(self):
        f = torch.zeros_like(self.x)
        force_noise = torch.randn_like(self.x) * self.noise
        f += force_noise

        # electrostatic repulsion
        electrical_force = torch.zeros_like(f)
        if self.repulsion != 0.0:
            if self.mac > 0:
                mass = self.weights #torch.ones_like(self.x[:, 0])
                qt = BarnesHutTree(self.x, mass, device=self.device, max_levels=self.max_levels)
                electrical_force = qt.traverse(self.x, mass, mac=self.mac, force_function=self.repulsion_function)
            else:
                diff = self.x.unsqueeze(0) - self.x.unsqueeze(1)
                m = (self.weights.unsqueeze(0) * self.weights.unsqueeze(1)).unsqueeze(2)
                electrical_force = torch.sum(m * (diff / ((torch.norm(diff, 2, dim=2, keepdim=True) ** 2) + 1e-5)), dim=0)
            electrical_force *= self.repulsion * self.spring_optimal_distance ** 2
        f += electrical_force

        # attraction
        attraction_force = torch.zeros_like(f)
        if self.spring_optimal_distance != 0.0:
            attraction_force = torch.zeros_like(f)
            diff = self.x[self.network.connections[:, 1]] - self.x[self.network.connections[:, 0]]
            dist = torch.norm(diff, 2, dim=1, keepdim=True)
            a_f = (diff * dist) / self.spring_optimal_distance
            a_f *= self.connection_weights.unsqueeze(1)
            attraction_force.scatter_add_(0, self.network.connections[:, 0:1].repeat([1, self.num_dim]), a_f)
            attraction_force.scatter_add_(0, self.network.connections[:, 1:2].repeat([1, self.num_dim]), -a_f)
            if self.attraction_normalization > 0.:
                attraction_force *= self.avg_connection_count / (1 + self.attraction_normalization * self.per_unit_connection_weight.unsqueeze(1))
        f += attraction_force

        # centering
        if self.centering > 0.:
            dist = torch.norm(self.x, 2, dim=1, keepdim=True)
            centering_force = -self.centering * (self.x / dist) * dist**2 * self.weights.unsqueeze(1)
            f += centering_force

        f_norm = torch.norm(f, 2, dim=1)
        energy = torch.sum(f ** 2)

        out_of_bound = f_norm > self.force_limit
        f[out_of_bound] = self.force_limit * f[out_of_bound] / f_norm[out_of_bound].unsqueeze(1)

        a = f / self.weights.unsqueeze(1)

        # velocity verlet integration
        self.v /= (1 + self.drag)
        self.v += 0.5 * (self.a + a) * self.step_size
        self.a = a
        x_update = self.movable.unsqueeze(1) * (self.v * self.step_size + 0.5 * self.a * self.step_size ** 2)

        is_nan = torch.isnan(x_update)
        if torch.sum(is_nan) > 0:
            print("x updates with nan:", is_nan.nonzero())
            x_update[is_nan] = torch.zeros(torch.sum(is_nan), device=self.x.device)

        self.x += x_update

        self.update_step_size(energy)
        self.energy = energy

    def simulate(self, num_steps=100, plot_interval=None):
        tic = time.time()
        for step in range(num_steps):
            self.simulation_step()
            if plot_interval is not None:
                if step % plot_interval == 0:
                    print("step", step, 'time per step:', (time.time() - tic) / plot_interval)
                    tic = time.time()
                    #self.plotting_thread = threading.Thread(target=self.plot)
                    #self.plotting_thread.daemon = True
                    #self.plotting_thread.start()
                    #self.plot()

    def update_step_size(self, new_energy):
        if new_energy < self.energy:
            self.energy_progress += 1
            if self.energy_progress >= 5:
                self.energy_progress = 0
                self.step_size /= self.step_discount_factor
        else:
            self.energy_progress = 0
            self.step_size *= self.step_discount_factor


class NetworkPlot:
    def __init__(self):
        # self.fig, self.ax = plt.subplots(1, 1)
        # self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.scatter = None
        self.lines = None


def animation_step(i, simulation, plot, plot_connections=True):
    net = simulation.network
    for _ in range(1):
        simulation.simulation_step()
    try:
        plot.scatter.remove()
        plot.lines.remove()
    except:
        pass
    if plot_connections:
        plot.lines = mc.LineCollection(net.line_data().cpu(), lw=0.5, alpha=0.2)
        plot.ax.add_collection(plot.lines)
    #print("energy:", simulation.energy)
    #print("step size:", simulation.step_size)
    pos = simulation.x.detach().cpu()
    plot.scatter = plot.ax.scatter(pos[:, 0], pos[:, 1], c=net.colors.cpu(), s=8.)
    plot.ax.autoscale()
    plot.fig.canvas.draw()
    return plot.lines, plot.scatter


def animate_simulation(simulation, plot, steps=50):
    ani = matplotlib.animation.FuncAnimation(plot.fig,
                                             animation_step,
                                             frames=steps,
                                             fargs=(simulation, plot, True),
                                             interval=50,
                                             repeat=False)
    return ani






