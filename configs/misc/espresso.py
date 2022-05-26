_base_ = './hyper_default.py'

expname = 'misc/espresso'
basedir = './logs/vrig_data'

data = dict(
    datadir='./espresso',
    dataset_type='hyper_dataset',
    white_bkgd=False,
)