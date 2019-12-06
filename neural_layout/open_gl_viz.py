import time
import numpy as np
import pickle
import vispy
import math
import time
import threading
import qtpy.QtWidgets
from vispy import gloo, app

vispy.use('PyQt5')

edges_vertex_shader = """
    uniform vec2 scaling;
    uniform mat3 transform;
    uniform vec2 shift;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;

    void main()
    {
        v_texcoord = texcoord;
        vec2 pos = (transform * vec3(position + shift, 0)).xy;
        pos = scaling * pos;
        gl_Position = vec4(pos, 0.0, 1.0);
    } """

edges_fragment_shader = """
    uniform sampler1D colormap;
    uniform float num_colors;
    uniform sampler2D texture;
    varying vec2 v_texcoord;

    void main()
    { 
      float value = texture2D(texture, v_texcoord).a;
      value = (1 + value * num_colors) / (num_colors + 2);
      gl_FragColor = texture1D(colormap, value).rgba;
    } """

nodes_vertex_shader = """
    uniform float scale;
    uniform float pix_size;
    uniform vec2 scaling;
    uniform mat3 transform;
    uniform vec2 shift;
    attribute vec2 center;
    attribute float radius;
    attribute float alpha;
    attribute float depth;
    varying vec2 v_center;
    varying float v_radius;
    varying float v_alpha;
    varying float v_depth;
    void main()
    {
        v_radius = radius;
        v_center = (transform * vec3(center + shift, 0)).xy;
        v_depth = depth;
        v_alpha = alpha;
        gl_PointSize = 2. + ceil(pix_size*radius*scale);
        gl_Position = vec4(v_center * scaling, 0.0, 1.0);
    } """

nodes_fragment_shader = """
    uniform float scale;
    uniform float pix_size;
    uniform vec2 offset;
    uniform sampler1D depth_colormap;
    uniform float num_depth_colors;
    varying vec2 v_center;
    varying float v_radius;
    varying float v_depth;
    varying float v_alpha;

    float circle(vec2 p, vec2 center, float radius)
    {
      return length(p - center) - radius;
    }

    void main()
    {
      vec2 p = 2 * (gl_FragCoord.xy - offset) / pix_size - 1;
      float a = v_alpha;
      float d = circle(p, v_center, v_radius*scale);
      if(d > 0.0) a = v_alpha * exp(-d*d*1000000.);
      float depth_value = (1 + v_depth * num_depth_colors) / (num_depth_colors + 2);
      gl_FragColor = vec4(texture1D(depth_colormap, depth_value).rgb, a);
    } """

focus_vertex_shader = """
    uniform float scale;
    uniform float radius;
    uniform float pix_size;
    uniform vec2 scaling;
    uniform mat3 transform;
    uniform vec2 shift;
    attribute vec2 center;
    attribute float depth;
    varying vec2 v_center;
    varying float v_radius;
    void main()
    {
        v_center = (transform * vec3(center + shift, 0)).xy;
        v_radius = radius;
        gl_PointSize = 2. + ceil(pix_size*radius*scale*2);
        gl_Position = vec4(v_center * scaling, 0.0, 1.0);
    } """

focus_fragment_shader = """
    uniform float scale;
    uniform float pix_size;
    uniform vec2 offset;
    uniform vec3 rgb;
    uniform float alpha;
    varying vec2 v_center;
    varying float v_radius;

    float circle(vec2 p, vec2 center, float radius)
    {
      return length(p - center) - radius;
    }

    void main()
    {
      vec2 p = 2 * (gl_FragCoord.xy - offset) / pix_size - 1;
      float a = alpha;
      float d = circle(p, v_center, v_radius*scale);
      if(d > 0.0) a = alpha * exp(-d*d*10000.);
      gl_FragColor = vec4(rgb, a);
    } """


class Visualizer(app.Canvas):
    def __init__(self, node_positions, edge_textures=None, node_weights=None, focus=None, animate=True, translate=True, draw_callback=None, size=(800, 600)):
        super().__init__(size=size, vsync=True)

        resolution = np.array([self.physical_size[0], self.physical_size[1]]).astype(np.float32)

        # transform transitions
        self._start_shift = np.float32([0, 0])
        self._start_scale = 1.
        self._start_theta = 0.
        self._current_shift = np.float32([0, 0])
        self._current_scale = 1.
        self._current_theta = 0.
        self._target_shift = np.float32([0, 0])
        self._target_scale = 1.
        self._target_theta = 0.
        self._transition_position = 1.
        self._current_transition_frame = 0
        self._transition_start_point = 0.
        self.transition_duration = None
        #self.fps = 60
        self.loop_duration = 4.
        self.sync_timepoint = 0.
        self.min_node_radius = 0.002
        self.node_radius_factor = 0.002
        self.weight_scaling_offset = 0.1
        self.node_alpha_factor = 0.3
        self.num_frames = node_positions.shape[0]

        self.scale_factor = 3.
        self.num_frames = node_positions.shape[0]
        self.animate = animate
        self.translate = translate
        self.draw_callback = draw_callback
        self.lock = threading.Lock()

        if edge_textures is None:
            edge_textures = np.zeros([self.num_frames, 2, 2], dtype=np.float32)
        self.edges_program = gloo.Program(edges_vertex_shader, edges_fragment_shader)
        self.edges_colors = np.array([[0., 0., 0., 1.], [1., 1., 1., 1.]]).astype(np.float32)
        self.edge_textures = edge_textures
        self.edges_program['texcoord'] = np.float32([[0, 1], [1, 1], [0, 0], [1, 0]])
        self.edges_program['texture'] = gloo.Texture2D(data=self.edge_textures[0], interpolation='linear', format='alpha')
        #self.edges_program['position'] = np.float32([[-1, 1], [1, 1], [-1, -1], [1, -1]])
        self.edges_program['position'] = np.float32([[-1, 1], [1, 1], [-1, -1], [1, -1]])
        #self.edges_program['resolution'] = resolution

        self.nodes_program = gloo.Program(nodes_vertex_shader, nodes_fragment_shader)
        self.node_colors = np.array([[1., 0., 0., 1.],
                                     [0.8, 0.8, 0., 1.],
                                     [0., 1., 0., 1.],
                                     [0., 0.8, 0.8, 1.],
                                     [0., 0., 1., 1.],
                                     [0.8, 0., 0.8, 1.],
                                     [1., 0., 0., 1.]]).astype(np.float32)
        if node_weights is None:
            node_weights = np.ones((self.num_frames, node_positions.shape[1])).astype(np.float32)
        self.node_weights = node_weights
        self.node_positions = node_positions
        self._node_positions_px = self.node_positions.copy()
        self.num_nodes = self.node_positions.shape[1]
        self.nodes_program['depth'] = np.linspace(0., 1., num=self.num_nodes).astype(np.float32)
        self.nodes_program['center'] = self.node_positions[0]
        # self.nodes_program['radius'] = 0.5 + self.node_weights[0] * 3.
        self.nodes_program['alpha'] = np.zeros(self.num_nodes).astype(np.float32) + 0.5
        #self.nodes_program['resolution'] = resolution
        scaling = np.float32([resolution[0] / max(resolution), resolution[1] / max(resolution)])
        self.nodes_program['scaling'] = scaling


        self.focus_program = gloo.Program(focus_vertex_shader, focus_fragment_shader)
        if focus is None:
            focus = np.zeros(self.node_positions.shape[1]) > 0.
        self.focus = focus
        self.focus_color = np.array([0.5, 1., 1., 0.5]).astype(np.float32)
        self.focus_program['center'] = self.node_positions[0, self.focus[0]]
        self.focus_program['radius'] = 0.005
        #self.focus_program['transform'] = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        # Enable blending
        gloo.set_state(blend=False, clear_color='black')

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

        self._current_frame = 0

        # self._timer = app.Timer(1/60., connect=self.update, start=True)
        # self.show()

    def set_new_node_positions(self, new_positions, new_weights=None, new_focus=None):
        old_num_nodes = self.node_positions.shape[1]
        num_nodes = new_positions.shape[1]
        with self.lock:
            self.node_positions = new_positions
            self._node_positions_px = new_positions.copy()
            if num_nodes != old_num_nodes:
                self.num_nodes = self.node_positions.shape[1]
                self.nodes_program['depth'] = np.linspace(0., 1., num=self.num_nodes).astype(np.float32)
            if new_weights is None and num_nodes != old_num_nodes:
                self.node_weights = np.ones((self.num_frames, num_nodes), dtype=np.float32)
            elif new_weights is not None:
                self.node_weights = new_weights
            if new_focus is None and num_nodes != old_num_nodes:
                self.focus = np.zeros(num_nodes) > 0.
            elif new_focus is not None:
                self.focus = new_focus

    @property
    def edges_colors(self):
        return self._edge_colors

    @edges_colors.setter
    def edges_colors(self, value):
        self._edge_colors = value
        value = value.astype(np.float32)
        value = np.concatenate([value[0:1, :], value, value[-2:-1, :]], axis=0)
        edges_colormap = gloo.Texture1D(data=value, interpolation='linear', wrapping='clamp_to_edge')
        self.edges_program['colormap'] = edges_colormap
        self.edges_program['num_colors'] = self._edge_colors.shape[0]

    @property
    def edge_textures(self):
        return self._edge_textures

    @edge_textures.setter
    def edge_textures(self, value):
        self._edge_textures = value
        #width = value.shape[1]
        #height = value.shape[2]
        self.edges_program['position'] = np.float32([[-1, 1], [1, 1], [-1, -1], [1, -1]])

    @property
    def node_colors(self):
        return self._node_colors

    @node_colors.setter
    def node_colors(self, value):
        self._node_colors = value
        value = value.astype(np.float32)
        value = np.concatenate([value[0:1, :], value, value[-2:-1, :]], axis=0)
        nodes_colormap = gloo.Texture1D(data=value, interpolation='linear', wrapping='clamp_to_edge')
        self.nodes_program['depth_colormap'] = nodes_colormap
        self.nodes_program['num_depth_colors'] = self._node_colors.shape[0]

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, value):
        if len(value.shape) == 1:
            flat_focus = value
            self._focus = np.repeat(value[np.newaxis, :], self.num_frames, axis=0)
        elif len(value.shape) == 2:
            self._focus = value
            flat_focus = np.sum(value, axis=0) > 0
        if not self.translate:
            return

        positions = self.node_positions[0, flat_focus]
        if len(positions) == 0.:
            width = 1.
            height = 1.
            mean_position = np.float32([0, 0])
        else:
            width = positions[:, 0].max() - positions[:, 0].min()
            height = positions[:, 1].max() - positions[:, 1].min()
            mean_position = np.mean(positions, axis=0) - 0.5
        size = math.sqrt(width * height)

        if sum(flat_focus) == 0:
            self._target_shift = np.float32([0, 0])
            self._target_scale = 1.
            self._target_theta = 0.

        self._target_shift = -2*mean_position
        self._target_scale = 1/(size + 1/self.scale_factor)
        self._target_theta = math.atan2(mean_position[0], mean_position[1]) * 1.

        self._start_shift = self._current_shift
        self._start_scale = self._current_scale
        self._start_theta = self._current_theta

        if abs(self._target_theta - self._current_theta) > abs(self._target_theta - (self._current_theta + np.pi * 2)):
            self._start_theta = self._current_theta + 2 * np.pi
        elif abs(self._target_theta - self._current_theta) > abs(self._target_theta - (self._current_theta - np.pi * 2)):
            self._start_theta = self._current_theta - np.pi * 2
        else:
            self._start_theta = self._current_theta

        self._current_transition_frame = 0
        self._transition_start_point = time.time()

    @property
    def focus_color(self):
        return self._focus_color

    @focus_color.setter
    def focus_color(self, value):
        self._focus_color = value.astype(np.float32)
        self.focus_program['rgb'] = self._focus_color[:3]
        self.focus_program['alpha'] = self._focus_color[3]

    def on_resize(self, event):
        width, height = event.physical_size
        size = min(width, height)
        offset_x = max(0, (width - size) // 2)
        offset_y = max(0, (height - size) // 2)
        # self.edges_program["position"] = np.float32([[offset_x, offset_y + size],
        #                                              [offset_x + size, offset_y + size],
        #                                              [offset_x, offset_y],
        #                                              [offset_x + size, offset_y]])
        self._node_positions_px[:, :, 0] = self.node_positions[:, :, 0] * size + offset_x
        self._node_positions_px[:, :, 1] = self.node_positions[:, :, 1] * size + offset_y
        gloo.set_viewport(0, 0, width, height)
        res = np.array([width, height]).astype(np.float32)
        pix_size = min(res)
        scaling = np.float32([pix_size / res[0], pix_size / res[1]])
        offset = 0.5 * (res - pix_size)

        #self.edges_program['resolution'] = res

        self.edges_program['scaling'] = scaling

        self.nodes_program['pix_size'] = pix_size
        self.nodes_program['offset'] = offset
        self.nodes_program['scaling'] = scaling

        self.focus_program['pix_size'] = pix_size
        self.focus_program['offset'] = offset
        self.focus_program['scaling'] = scaling

    def on_draw(self, event):
        self.lock.acquire()

        if self.animate:
            current_frame = ((time.time() - self.sync_timepoint) / self.loop_duration) * self.num_frames
            current_frame = int(current_frame) % self.num_frames
        else:
            current_frame = 0

        self.edges_program['texture'] = self.edge_textures[current_frame]
        self.nodes_program['center'] = self.node_positions[current_frame] * 2. - 1.
        sqrt_weight = np.sqrt(np.abs(self.node_weights[current_frame]) + self.weight_scaling_offset)
        self.nodes_program['radius'] = self.min_node_radius + sqrt_weight * self.node_radius_factor
        self.nodes_program['alpha'] = np.clip(sqrt_weight * self.node_alpha_factor, 0., 0.5)
        self.focus_program['center'] = self.node_positions[current_frame][self.focus[current_frame]] * 2. - 1.
        #self.focus_program['radius'] = 0.002 + self.node_weights[self.current_frame] * 0.005

        if self.transition_duration is not None:
            transition_position = (time.time() - self._transition_start_point) / self.transition_duration
            # self._transition_position = self._current_transition_frame / self.transition_frames
            # self._current_transition_frame += 1.
        else:
            transition_position = 1.
        p = self.interpolation_function(transition_position)
        q = 1 - p
        self._current_shift = q * self._start_shift + p * self._target_shift
        self._current_scale = q * self._start_scale + p * self._target_scale
        self._current_theta = q * self._start_theta + p * self._target_theta
        transform = self.calc_affine_matrix([0., 0.,
                                             self._current_scale,
                                             self._current_scale,
                                             0., 0.,
                                             self._current_theta])

        #transform = np.float32([[1., 0.0, ], [0, 1, 0], [0, 0, 0]])
        self.edges_program['transform'] = transform
        self.edges_program['shift'] = self._current_shift
        self.nodes_program['transform'] = transform
        self.nodes_program['shift'] = self._current_shift
        self.focus_program['transform'] = transform
        self.focus_program['shift'] = self._current_shift
        self.nodes_program['scale'] = self._current_scale
        self.focus_program['scale'] = self._current_scale
        gloo.clear()
        gloo.set_state(blend=True, clear_color='black', blend_func=('dst_alpha', 'one_minus_dst_alpha'))
        self.edges_program.draw('triangle_strip')
        gloo.set_state(blend=True, clear_color='black',
                       blend_func=('src_alpha', 'dst_alpha'))
        self.nodes_program.draw('points')
        gloo.set_state(blend=True, clear_color='black',
                       blend_func=('src_alpha', 'dst_alpha'))
        self.focus_program.draw('points')
        if self.draw_callback is not None:
            self.draw_callback()
        self.lock.release()
        # if self.animate:
        #     self._current_frame = (self._current_frame + 1) % self.num_frames

    @staticmethod
    def calc_affine_matrix(p):
        # input: numpy array with
        #   0         1         2        3        4        5        6
        # center_x, center_y, scale_x, scale_y, shift_x, shift_y, theta
        cos_th = math.cos(p[6])
        sin_th = math.sin(p[6])

        a = cos_th * p[2]
        b = sin_th * p[2]
        c = p[4] - p[0] * a - p[1] * b
        d = -sin_th * p[3]
        e = cos_th * p[3]
        f = p[5] - p[0] * d - p[1] * e

        return np.float32([[a, b, c], [d, e, f], [0., 0., 1.]])

    @staticmethod
    def interpolation_function(value):
        value = min(max(0., value), 1.)
        return value ** 2 * (3 - value * 2)
