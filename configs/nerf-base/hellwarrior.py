_base_ = './default.py'

expname = 'base/dnerf_hellwarrior-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data_dnerf/hellwarrior',
    dataset_type='dnerf',
    white_bkgd=True,
)

