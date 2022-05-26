_base_ = './hyper_default.py'

expname = 'vrig/base-peel-banana'
basedir = './logs/vrig_data'

data = dict(
    datadir='./vrig-peel-banana',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)