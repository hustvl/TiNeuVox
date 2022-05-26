_base_ = './hyper_default.py'

expname = 'vrig/base-broom'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig_dataset/broom2',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)