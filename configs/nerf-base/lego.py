_base_ = './default.py'

expname = 'base/dnerf_lego-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data_dnerf/lego',
    dataset_type='dnerf',
    white_bkgd=True,
)

