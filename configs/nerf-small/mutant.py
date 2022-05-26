_base_ = './default.py'

expname = 'small/dnerf_mutant-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data_dnerf/mutant',
    dataset_type='dnerf',
    white_bkgd=True,
)

