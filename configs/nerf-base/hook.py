_base_ = './default.py'

expname = 'base/dnerf_hook-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data_dnerf/hook',
    dataset_type='dnerf',
    white_bkgd=True,
)

