from neural_layout.network_architectures import resnet18_1d_network
from neural_layout.layout_calculation import LayoutCalculation
import imageio
import torch
import numpy as np

net = resnet18_1d_network()
output_video_file = 'resnet18_1d_layout.mp4'
output_positions_file = 'resnet18_1d_layout'  # output .npy file containing the positions of the neurons

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'

if output_video_file is not None:
    video_writer = imageio.get_writer(output_video_file, fps=30)
else:
    video_writer = None

layout_calculation = LayoutCalculation(net=net, video_writer=video_writer, device=dev, size=(800, 800))

layout_calculation.range_gamma = 0.5  # how quickly the view size adapts to new neuron positions
layout_calculation.centering = 0.  # initial centering force
layout_calculation.additional_centering_per_level = 1.  # progressive centering force
layout_calculation.max_centering = 20.  # maximum centering force

layout_calculation.viz.scale_factor = 100.
layout_calculation.viz.focus = np.zeros(layout_calculation.viz.node_positions.shape[1]) > 0.  # no neurons are in focus

# merge neighbouring neurons
for i in range(8):
    net = net.collapse_layers(factor=2, dimension=0)
    net.to(dev)
for i in range(8):
    net = net.collapse_layers(factor=2, dimension=1)
    net.to(dev)

layout_calculation.net = net
layout_calculation.plot_connections = True

layout_positions = layout_calculation.start_simulation()
np.save('resnet18_1d_layout_positions_2', layout_positions)
video_writer.close()