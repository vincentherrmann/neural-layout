from neural_layout.network_graph import Network


def vgg16_network():
    net = Network()
    net.add_layer('input', [1, 224, 224])
    net.add_layer('l1', [1, 224, 224])
    net.add_layer('l2', [1, 224, 224])
    net.add_layer('l3', [2, 112, 112])
    net.add_layer('l4', [2, 112, 112])
    net.add_layer('l5', [4, 56, 56])
    net.add_layer('l6', [4, 56, 56])
    net.add_layer('l7', [4, 56, 56])
    net.add_layer('l8', [8, 28, 28])
    net.add_layer('l9', [8, 28, 28])
    net.add_layer('l10', [8, 28, 28])
    net.add_layer('l11', [8, 14, 14])
    net.add_layer('l12', [8, 14, 14])
    net.add_layer('l13', [8, 14, 14])
    net.add_layer('l14', [64, 1, 1])
    net.add_layer('l15', [64, 1, 1])
    net.add_layer('output', [16, 1, 1])

    net.add_conv2d_connections('input', 'l1', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l1', 'l2', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))

    net.add_conv2d_connections('l2', 'l3', stride=(2, 2),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l3', 'l4', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))

    net.add_conv2d_connections('l4', 'l5', stride=(2, 2),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l5', 'l6', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l6', 'l7', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))

    net.add_conv2d_connections('l7', 'l8', stride=(2, 2),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l8', 'l9', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l9', 'l10', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))

    net.add_conv2d_connections('l10', 'l11', stride=(2, 2),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l11', 'l12', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))
    net.add_conv2d_connections('l12', 'l13', stride=(1, 1),
                               kernel_size=(3, 3), padding=(1, 1, 1, 1))

    net.add_full_connections('l13', 'l14')
    net.add_full_connections('l14', 'l15')
    net.add_full_connections('l15', 'output')

    return net

def vgg16_1d_network():
    net = Network()
    net.add_layer('input', [1, 224])
    net.add_layer('l1', [8, 224])
    net.add_layer('l2', [8, 224])
    net.add_layer('l3', [16, 112])
    net.add_layer('l4', [16, 112])
    net.add_layer('l5', [32, 56])
    net.add_layer('l6', [32, 56])
    net.add_layer('l7', [32, 56])
    net.add_layer('l8', [64, 28])
    net.add_layer('l9', [64, 28])
    net.add_layer('l10', [64, 28])
    net.add_layer('l11', [64, 14])
    net.add_layer('l12', [64, 14])
    net.add_layer('l13', [64, 14])
    net.add_layer('l14', [512, 1])
    net.add_layer('l15', [512, 1])
    net.add_layer('output', [128, 1])

    net.add_conv1d_connections('input', 'l1', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l1', 'l2', stride=1,
                               kernel_size=3, padding=(1, 1))

    net.add_conv1d_connections('l2', 'l3', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l3', 'l4', stride=1,
                               kernel_size=3, padding=(1, 1))

    net.add_conv1d_connections('l4', 'l5', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l5', 'l6', stride=1,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l6', 'l7', stride=1,
                               kernel_size=3, padding=(1, 1))

    net.add_conv1d_connections('l7', 'l8', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l8', 'l9', stride=1,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l9', 'l10', stride=1,
                               kernel_size=3, padding=(1, 1))

    net.add_conv1d_connections('l10', 'l11', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l11', 'l12', stride=1,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('l12', 'l13', stride=1,
                               kernel_size=3, padding=(1, 1))

    net.add_full_connections('l13', 'l14')
    net.add_full_connections('l14', 'l15')
    net.add_full_connections('l15', 'output')

    return net

def resnet18_1d_network():
    net = Network()
    net.add_layer('input', [1, 224])
    net.add_layer('conv1', [64, 112])

    net.add_layer('conv2_1_a', [64, 56])
    net.add_layer('conv2_1_b', [64, 56])
    net.add_layer('conv2_2_a', [64, 56])
    net.add_layer('conv2_2_b', [64, 56])

    net.add_layer('conv3_1_a', [128, 28])
    net.add_layer('conv3_1_b', [128, 28])
    net.add_layer('conv3_2_a', [128, 28])
    net.add_layer('conv3_2_b', [128, 28])

    net.add_layer('conv4_1_a', [256, 14])
    net.add_layer('conv4_1_b', [256, 14])
    net.add_layer('conv4_2_a', [256, 14])
    net.add_layer('conv4_2_b', [256, 14])

    net.add_layer('conv5_1_a', [512, 7])
    net.add_layer('conv5_1_b', [512, 7])
    net.add_layer('conv5_2_a', [512, 7])
    net.add_layer('conv5_2_b', [512, 7])

    net.add_layer('average_pool', [512, 1])
    net.add_layer('fully_connected', [1000, 1])

    net.add_conv1d_connections('input', 'conv1', stride=2,
                               kernel_size=7, padding=(3, 3))

    # conv2
    net.add_conv1d_connections('conv1', 'conv2_1_a', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv2_1_a', 'conv2_1_b',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv1', 'conv2_1_b', stride=2,
                               kernel_size=1)

    net.add_conv1d_connections('conv2_1_b', 'conv2_2_a',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv2_2_a', 'conv2_2_b',
                               kernel_size=3, padding=(1, 1))
    net.add_one_to_one_connections('conv2_1_b', 'conv2_2_b')

    # conv3
    net.add_conv1d_connections('conv2_2_b', 'conv3_1_a', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv3_1_a', 'conv3_1_b',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv2_2_b', 'conv3_1_b', stride=2,
                               kernel_size=1)

    net.add_conv1d_connections('conv3_1_b', 'conv3_2_a',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv3_2_a', 'conv3_2_b',
                               kernel_size=3, padding=(1, 1))
    net.add_one_to_one_connections('conv3_1_b', 'conv3_2_b')

    # conv4
    net.add_conv1d_connections('conv3_2_b', 'conv4_1_a', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv4_1_a', 'conv4_1_b',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv3_2_b', 'conv4_1_b', stride=2,
                               kernel_size=1)

    net.add_conv1d_connections('conv4_1_b', 'conv4_2_a',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv4_2_a', 'conv4_2_b',
                               kernel_size=3, padding=(1, 1))
    net.add_one_to_one_connections('conv4_1_b', 'conv4_2_b')

    # conv5
    net.add_conv1d_connections('conv4_2_b', 'conv5_1_a', stride=2,
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv5_1_a', 'conv5_1_b',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv4_2_b', 'conv5_1_b', stride=2,
                               kernel_size=1)

    net.add_conv1d_connections('conv5_1_b', 'conv5_2_a',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('conv5_2_a', 'conv5_2_b',
                               kernel_size=3, padding=(1, 1))
    net.add_one_to_one_connections('conv5_1_b', 'conv5_2_b')

    net.add_conv1d_connections('conv5_2_b', 'average_pool', kernel_size=7)
    net.add_full_connections('average_pool', 'fully_connected')
    return net

def scalogram_resnet_network():
    net = Network()
    net.add_layer('scalogram', [2, 292])

    net.add_layer('scalogram_block_0_main_conv_1', [32, 228])
    net.add_layer('scalogram_block_0_main_conv_2', [32, 114])

    net.add_layer('scalogram_block_1_main_conv_1', [32, 114])
    net.add_layer('scalogram_block_1_main_conv_2', [32, 114])

    net.add_layer('scalogram_block_2_main_conv_1', [64, 82])
    net.add_layer('scalogram_block_2_main_conv_2', [64, 41])

    net.add_layer('scalogram_block_3_main_conv_1', [64, 41])
    net.add_layer('scalogram_block_3_main_conv_2', [64, 41])

    net.add_layer('scalogram_block_4_main_conv_1', [128, 26])
    net.add_layer('scalogram_block_4_main_conv_2', [128, 13])

    net.add_layer('scalogram_block_5_main_conv_1', [128, 13])
    net.add_layer('scalogram_block_5_main_conv_2', [128, 13])

    net.add_layer('scalogram_block_6_main_conv_1', [256, 5])
    net.add_layer('scalogram_block_6_main_conv_2', [256, 5])

    net.add_layer('scalogram_block_7_main_conv_1', [512, 3])
    net.add_layer('scalogram_block_7_main_conv_2', [512, 1])

    net.add_layer('ar_block_0', [512, 1])
    net.add_layer('ar_block_1', [512, 1])
    net.add_layer('ar_block_2', [512, 1])
    net.add_layer('ar_block_3', [512, 1])
    net.add_layer('ar_block_4', [256, 1])
    net.add_layer('ar_block_5', [256, 1])
    net.add_layer('ar_block_6', [256, 1])
    net.add_layer('ar_block_7', [256, 1])
    net.add_layer('ar_block_8', [256, 1])

    # Encoder
    # BLOCK 0
    net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_1',
                               kernel_size=65)
    net.add_conv1d_connections('scalogram_block_0_main_conv_1', 'scalogram_block_0_main_conv_2',
                               kernel_size=3, stride=2, padding=(1, 1))
    net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 1
    net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_1_main_conv_1', 'scalogram_block_1_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_2',
                               kernel_size=1)

    # BLOCK 2
    net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_1',
                               kernel_size=33)
    net.add_conv1d_connections('scalogram_block_2_main_conv_1', 'scalogram_block_2_main_conv_2',
                               kernel_size=3, stride=2, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 3
    net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_3_main_conv_1', 'scalogram_block_3_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_2',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_1',
                               kernel_size=16)
    net.add_conv1d_connections('scalogram_block_4_main_conv_1', 'scalogram_block_4_main_conv_2',
                               kernel_size=3, stride=2, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 5
    net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_5_main_conv_1', 'scalogram_block_5_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_2',
                               kernel_size=1)

    # BLOCK 6
    net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_1',
                               kernel_size=9)
    net.add_conv1d_connections('scalogram_block_6_main_conv_1', 'scalogram_block_6_main_conv_2',
                               kernel_size=3, stride=1, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 7
    net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_1',
                               kernel_size=3)
    net.add_conv1d_connections('scalogram_block_7_main_conv_1', 'scalogram_block_7_main_conv_2',
                               kernel_size=3)
    net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_2',
                               kernel_size=1)

    # Autoregressive model
    # BLOCK 0
    net.add_conv1d_connections('scalogram_block_7_main_conv_2', 'ar_block_0',
                               kernel_size=1)

    # BLOCK 1
    net.add_conv1d_connections('ar_block_0', 'ar_block_1',
                               kernel_size=1)

    # BLOCK 2
    net.add_conv1d_connections('ar_block_1', 'ar_block_2',
                               kernel_size=1)

    # BLOCK 3
    net.add_conv1d_connections('ar_block_2', 'ar_block_3',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('ar_block_3', 'ar_block_4',
                               kernel_size=1)

    # BLOCK 5
    net.add_conv1d_connections('ar_block_4', 'ar_block_5',
                               kernel_size=1)

    # BLOCK 3
    net.add_conv1d_connections('ar_block_5', 'ar_block_6',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('ar_block_6', 'ar_block_7',
                               kernel_size=1)

    # BLOCK 5
    net.add_conv1d_connections('ar_block_7', 'ar_block_8',
                               kernel_size=1)

    # scoring
    net.add_conv1d_connections('ar_block_8', 'scalogram_block_7_main_conv_2',
                               kernel_size=1)

    return net


def scalogram_resnet_network_smaller():
    net = Network()
    net.add_layer('scalogram', [2, 216])

    net.add_layer('scalogram_block_0_main_conv_1', [8, 108])
    net.add_layer('scalogram_block_0_main_conv_2', [8, 84])

    net.add_layer('scalogram_block_1_main_conv_1', [16, 84])
    net.add_layer('scalogram_block_1_main_conv_2', [16, 84])

    net.add_layer('scalogram_block_2_main_conv_1', [32, 84])
    net.add_layer('scalogram_block_2_main_conv_2', [32, 60])

    net.add_layer('scalogram_block_3_main_conv_1', [64, 60])
    net.add_layer('scalogram_block_3_main_conv_2', [64, 60])

    net.add_layer('scalogram_block_4_main_conv_1', [128, 30])
    net.add_layer('scalogram_block_4_main_conv_2', [128, 6])

    net.add_layer('scalogram_block_5_main_conv_1', [256, 6])
    net.add_layer('scalogram_block_5_main_conv_2', [256, 6])

    net.add_layer('scalogram_block_6_main_conv_1', [512, 6])
    net.add_layer('scalogram_block_6_main_conv_2', [512, 3])

    net.add_layer('scalogram_block_7_main_conv_1', [512, 3])
    net.add_layer('scalogram_block_7_main_conv_2', [512, 1])

    net.add_layer('ar_block_0', [512, 1])
    net.add_layer('ar_block_1', [512, 1])
    net.add_layer('ar_block_2', [512, 1])
    net.add_layer('ar_block_3', [512, 1])
    net.add_layer('ar_block_4', [256, 1])
    net.add_layer('ar_block_5', [256, 1])
    net.add_layer('ar_block_6', [256, 1])
    net.add_layer('ar_block_7', [256, 1])
    net.add_layer('ar_block_8', [256, 1])

    # Encoder
    # BLOCK 0
    net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_1',
                               kernel_size=3, stride=2, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_0_main_conv_1', 'scalogram_block_0_main_conv_2',
                               kernel_size=25)
    net.add_conv1d_connections('scalogram', 'scalogram_block_0_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 1
    net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_1_main_conv_1', 'scalogram_block_1_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_0_main_conv_2', 'scalogram_block_1_main_conv_2',
                               kernel_size=1)

    # BLOCK 2
    net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_2_main_conv_1', 'scalogram_block_2_main_conv_2',
                               kernel_size=25)
    net.add_conv1d_connections('scalogram_block_1_main_conv_2', 'scalogram_block_2_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 3
    net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_3_main_conv_1', 'scalogram_block_3_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_2_main_conv_2', 'scalogram_block_3_main_conv_2',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_1',
                               kernel_size=3, stride=2, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_4_main_conv_1', 'scalogram_block_4_main_conv_2',
                               kernel_size=25)
    net.add_conv1d_connections('scalogram_block_3_main_conv_2', 'scalogram_block_4_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 5
    net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_5_main_conv_1', 'scalogram_block_5_main_conv_2',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_4_main_conv_2', 'scalogram_block_5_main_conv_2',
                               kernel_size=1)

    # BLOCK 6
    net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_6_main_conv_1', 'scalogram_block_6_main_conv_2',
                               kernel_size=4)
    net.add_conv1d_connections('scalogram_block_5_main_conv_2', 'scalogram_block_6_main_conv_2',
                               kernel_size=1, stride=2)

    # BLOCK 7
    net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_1',
                               kernel_size=3, padding=(1, 1))
    net.add_conv1d_connections('scalogram_block_7_main_conv_1', 'scalogram_block_7_main_conv_2',
                               kernel_size=3)
    net.add_conv1d_connections('scalogram_block_6_main_conv_2', 'scalogram_block_7_main_conv_2',
                               kernel_size=1)

    # Autoregressive model
    # BLOCK 0
    net.add_conv1d_connections('scalogram_block_7_main_conv_2', 'ar_block_0',
                               kernel_size=1)

    # BLOCK 1
    net.add_conv1d_connections('ar_block_0', 'ar_block_1',
                               kernel_size=1)

    # BLOCK 2
    net.add_conv1d_connections('ar_block_1', 'ar_block_2',
                               kernel_size=1)

    # BLOCK 3
    net.add_conv1d_connections('ar_block_2', 'ar_block_3',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('ar_block_3', 'ar_block_4',
                               kernel_size=1)

    # BLOCK 5
    net.add_conv1d_connections('ar_block_4', 'ar_block_5',
                               kernel_size=1)

    # BLOCK 3
    net.add_conv1d_connections('ar_block_5', 'ar_block_6',
                               kernel_size=1)

    # BLOCK 4
    net.add_conv1d_connections('ar_block_6', 'ar_block_7',
                               kernel_size=1)

    # BLOCK 5
    net.add_conv1d_connections('ar_block_7', 'ar_block_8',
                               kernel_size=1)

    # scoring
    net.add_conv1d_connections('ar_block_8', 'scalogram_block_7_main_conv_2',
                               kernel_size=1)

    return net