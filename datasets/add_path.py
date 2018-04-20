import socket

if socket.gethostname()[0:6] == 'napoli':
    data_root = '/scr/kuanfang/Datasets/MOT_track'
    data_root = './data'
elif socket.gethostname() == 'capri16.stanford.edu':
    data_root = '/capri16/Datasets/MOT_track'
else:
    raise ValueError('Unrecognized hostname {}.'.format(socket.gethostname()))

# data_root = './data'

