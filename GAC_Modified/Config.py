import logging
from pathlib import Path
from datetime import datetime


class Hparams(object):
    data_name = "toronto_3d"

    root = "/home/hope/rebuttalPT/4096/Toronto_3D"  # todo: check to make sure

    num_classes = 8
    dropout = 0.5
    learning_rate = 0.01
    alpha = 0.2
    single_nsample = 32
    radius = [0.3, 0.5, 0.8, 1.6]  # one per layer
    npoint = [1024, 256, 128, 64]  # one per layer
    group_all = False
    class_weights = [100148711, 4064402, 12949350, 62706871, 2344352, 1991686, 24077355, 834457]
    model_name = 'backbone_with_res' # 'backbone' 'backbone_with_pna' 'backbone_with_res' 'backbone_with_pna_combine'

    '''class str:num dict n viceversa'''
    classes = ['ground', 'road_markings', 'natural', 'building', 'utility_line', 'pole', 'car', 'fence']
    class2label = {cls: i for i, cls in enumerate(classes)}  # keys:value  category:label  str:num
    

    # pna params
    aggregators = "mean max min std"
    scalers = "identity amplification attenuation"
    tot_aggs = len([agg for agg in aggregators.split(' ')])
    tot_scalers = len([scale for scale in scalers.split(' ')])
    threshold = 0.1
    post_trans = True


    # multi-scale nbrhood parameters
    nscales = 2 # that is len(multi_nsamples) to deploy
    multi_nsamples = [20, 44, 60] 
    # todo: reason, pick the largest nbrhood n get the smaller ones from it
    max_radius = [2.0, 4.0, 6.0]  # one per multi-dilated layer

    '''CREATE DIR'''
    experiment_dir = Path('experiment/')  # todo: check to make sure
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)



def start_logger(Hparams):
    """Set the logger to log info in terminal and file `log_path`.
    i.e.,'model_dir/tr_log'
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # logging to file
    file_handler = logging.FileHandler(str(Hparams.log_dir) + '/GAC_Modified.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('\t\t %(message)s'))
    logger.addHandler(stream_handler)

    logging.info('Dataset name: %s' % (Hparams.data_name))
    logging.info('Name of model implemented: %s' % (Hparams.model_name))
    logging.info('Dropout rate: %f' % (Hparams.dropout))
