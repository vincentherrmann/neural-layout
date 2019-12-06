import torch


def gravity_function(m1, m2, difference, distance):
    return (m1 * m2 / distance**3).unsqueeze(1) * difference


def electrostatic_function(m1, m2, difference, distance):
    return (m1 * m2 / distance**2).unsqueeze(1) * difference


def energy_function(m1, m2, difference, distance):
    return -(m1 * m2 / distance**3).unsqueeze(1).repeat(1, 3)

class BarnesHutTree(object):
    def __init__(self, pos, mass, max_levels=100, device='cpu'):
        super().__init__()
        self.device = device

        self.num_levels = 0
        self.max_levels = max_levels

        self.num_dim = pos.shape[1]
        self.num_o = 2**self.num_dim

        min_val = torch.min(pos) - 1e-4
        max_val = torch.max(pos) + 1e-4
        self.size = max_val - min_val

        norm_pos = (pos - min_val.unsqueeze(0)) / self.size.unsqueeze(0)  # normalized position of all points

        # level-wise tree parameters (list index is the corresponding level)
        self.node_mass = []
        self.center_of_mass = []
        self.is_end_node = []
        self.node_indexing = []

        point_nodes = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)  # node in which each point falls on the current level
        num_nodes = 1

        while True:
            self.num_levels += 1
            num_divisions = 2**self.num_levels

            # calculate the orthant in which each point falls
            point_orthant = torch.floor(norm_pos * num_divisions).long()
            point_orthant = (point_orthant % 2) * (2**torch.arange(self.num_dim, device=self.device).unsqueeze(0))
            point_orthant = torch.sum(point_orthant, dim=1)

            # calculate node indices from point orthants
            point_nodes *= self.num_o
            point_nodes += point_orthant

            # calculate total mass of each section
            node_mass = torch.zeros(num_nodes * self.num_o, device=self.device)
            node_mass.scatter_add_(0, point_nodes, mass)

            # calculate center of mass of each node
            node_com = torch.zeros(num_nodes * self.num_o, self.num_dim, device=self.device)
            for d in range(self.num_dim):
                node_com[:, d].scatter_add_(0, point_nodes, pos[:, d] * mass)
            node_com /= node_mass.unsqueeze(1)

            # determine if node is end node
            point_is_continued = node_mass[point_nodes] > mass  # only points that are not the only ones in their node are passed on to the next level
            end_nodes = point_nodes[point_is_continued == 0]  # nodes with only one point are end nodes
            is_end_node = torch.zeros(num_nodes * self.num_o, device=self.device, dtype=torch.bool)
            is_end_node[end_nodes] = 1

            node_is_continued = node_mass > 0.
            non_empty_nodes = node_is_continued.nonzero().squeeze(1)  # indices of non-empty nodes
            num_nodes = non_empty_nodes.shape[0]

            # create new node indexing: only non-empty nodes have positive indices, end nodes have the index -1
            node_indexing = self.create_non_empty_node_indexing(non_empty_nodes, node_mass.shape[0], self.num_o)

            # only pass on nodes that are continued
            is_end_node = is_end_node[node_is_continued]
            node_mass = node_mass[node_is_continued]
            node_com = node_com[node_is_continued, :]

            self.node_mass.append(node_mass)
            self.center_of_mass.append(node_com)
            self.node_indexing.append(node_indexing)
            self.is_end_node.append(is_end_node)

            # update the node index of each point
            point_nodes = node_indexing[point_nodes / self.num_o, point_nodes % self.num_o]

            # discard points in end nodes
            pos = pos[point_is_continued]
            mass = mass[point_is_continued]
            point_nodes = point_nodes[point_is_continued]
            norm_pos = norm_pos[point_is_continued]

            if torch.sum(point_is_continued) < 1:
                break
            if self.num_levels >= self.max_levels:
                num_points_in_nodes = torch.zeros_like(node_mass, dtype=torch.long)
                num_points_in_nodes.scatter_add_(0, point_nodes, torch.ones_like(mass, dtype=torch.long))
                max_points_in_node = torch.max(num_points_in_nodes)
                non_empty_nodes, index_of_point = torch.unique(point_nodes, return_inverse=True)
                node_index_of_point = non_empty_nodes[index_of_point]
                scatter_indices = torch.arange(node_index_of_point.shape[0], device=self.device) % max_points_in_node
                point_order = torch.argsort(node_index_of_point)
                node_indexing = torch.zeros(num_nodes, max_points_in_node,
                                            dtype=torch.long, device=self.device) - 1
                node_indexing[node_index_of_point[point_order], scatter_indices] = torch.arange(node_index_of_point.shape[0], device=self.device)[point_order]
                self.node_mass.append(mass)
                self.center_of_mass.append(pos)
                self.node_indexing.append(node_indexing)
                self.is_end_node.append(torch.ones_like(mass, dtype=torch.bool))
                #print("too many levels!")
                break

    def create_non_empty_node_indexing(self, non_empty_nodes, num_nodes, refinement_factor):
        # create new node indexing: only non-empty nodes have positive indices, end nodes have the index -1
        new_indices = torch.arange(non_empty_nodes.shape[0], device=self.device)
        node_indexing = torch.zeros(num_nodes // refinement_factor, refinement_factor,
                                    dtype=torch.long, device=self.device) - 1
        node_indexing[non_empty_nodes // refinement_factor, non_empty_nodes % refinement_factor] = new_indices
        return node_indexing

    def traverse(self, x, m, mac=0.7, force_function=gravity_function):
        force = torch.zeros_like(x)

        # pair each point with all nodes of the first level
        pairs_o = torch.cat([torch.arange(x.shape[0],
                                          dtype=torch.long,
                                          device=self.device).unsqueeze(1).repeat(1, self.num_o).view(-1, 1),
                             torch.arange(self.num_o,
                                          dtype=torch.long,
                                          device=self.device).unsqueeze(1).repeat(x.shape[0], 1)],
                            dim=1)

        refine = torch.stack([torch.arange(x.shape[0], dtype=torch.long, device=self.device),
                              torch.zeros(x.shape[0], dtype=torch.long, device=self.device)], dim=1)

        for l in range(len(self.node_indexing)):
            refinement_factor = self.node_indexing[l].shape[1]
            refine[:, 1] *= refinement_factor
            refine = refine.unsqueeze(1).repeat(1, refinement_factor, 1)
            refine[:, :, 1] = refine[:, :, 1] + torch.arange(refinement_factor, dtype=torch.long,
                                                             device=self.device).unsqueeze(0)
            pairs_o = refine.view(-1, 2)

            indexing = self.node_indexing[l]
            pairs = pairs_o.clone()

            # adjust indexing of the nodes
            pairs[:, 1] = indexing[pairs_o[:, 1] / refinement_factor, pairs_o[:, 1] % refinement_factor]
            # remove nodes with index -1
            pairs = pairs[pairs[:, 1] >= 0, :]

            this_com = self.center_of_mass[l][pairs[:, 1], :]
            this_mass = self.node_mass[l][pairs[:, 1]]

            diff = x[pairs[:, 0], :] - this_com
            dist = torch.norm(diff, 2, dim=1)
            if l < self.num_levels:
                section_size = self.size / 2 ** (l + 1)
            else:
                section_size = 0.
                #print("truncation level pairs:", pairs.shape[0])
            d2r = section_size / dist

            relative_weight_difference = torch.abs((m[pairs[:, 0]] - this_mass) * this_mass)
            different_mass = relative_weight_difference > 0.01

            mac_accept = (d2r < mac)
            end_node = self.is_end_node[l][pairs[:, 1]]
            accept = torch.max(mac_accept, end_node * (dist > 1e-5*section_size))

            this_f = force_function(m1=this_mass[accept],
                                    m2=m[pairs[:, 0]][accept],
                                    difference=diff[accept],
                                    distance=dist[accept])
            force[:, 0].scatter_add_(0, pairs[:, 0][accept], this_f[:, 0])
            force[:, 1].scatter_add_(0, pairs[:, 0][accept], this_f[:, 1])

            # get pairs that were not accepted
            refine = pairs[(accept == 0).nonzero(), :].squeeze(1)

        return force









