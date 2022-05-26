_base_ = './default.py'

expname = 'small/dnerf_jumpingjacks-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data_dnerf/jumpingjacks',
    dataset_type='dnerf',
    white_bkgd=True,
)

