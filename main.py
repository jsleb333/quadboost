import torch
import logging

from quadboost import QuadBoostMHCR
from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from weak_learner import *
from callbacks import ModelCheckpoint, CSVLogger, BreakOnZeroRiskCallback
from mnist_dataset import MNISTDataset
from utils import parse, timed


@timed
@parse
def main(m=60_000, dataset='mnist', center=True, reduce=True, encodings='onehot', wl='rf', max_round=1000, patience=1000, resume=0, n_jobs=1, max_n_leaves=4, n_filters=10, ks=11, locality=5, init_filters='from_bank', bank_ratio=.05, fn='', seed=42):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    ### Data loading
    mnist = MNISTDataset.load(dataset+'.pkl')
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=center, reduce=reduce)
    idx = np.arange(m)
    if seed:
        np.random.shuffle(idx)
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    logging.info(f'Loaded dataset: {dataset} (center: {center}, reduce: {reduce})')

    ### Choice of encoder
    if encodings == 'onehot':
        encoder = OneHotEncoder(Ytr)
    elif encodings == 'allpairs':
        encoder = AllPairsEncoder(Ytr)
    else:
        encoder = LabelEncoder.load_encodings(encodings)
        if all(label.isdigit() for label in encoder.labels_encoding):
            encoder = LabelEncoder({int(label):encoding for label, encoding in encoder.labels_encoding.items()})
    logging.info(f'Encoding: {encodings}')

    filename = f'd={dataset}-e={encodings}-wl={wl}'

    ### Choice of weak learner
    kwargs = {}
    if wl == 'ds' or wl == 'decision-stump':
        weak_learner = MulticlassDecisionStump()
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs
        
    elif wl == 'dt' or wl == 'decision-tree':
        weak_learner = MulticlassDecisionTree(max_n_leaves=max_n_leaves)
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs
        filename += f'{max_n_leaves}'
        
    elif wl == 'ridge':
        weak_learner = WLThresholdedRidge(threshold=.5)
        
    elif wl == 'rf' or wl == 'random_filters':
        filename += f'-nf={n_filters}-ks={ks}-{init_filters}'
            
        filter_bank = None
        if init_filters == 'from_bank':
            if 0 < bank_ratio < 1:
                bank_size = int(m*bank_ratio)
                filter_bank = Xtr[:bank_size]
                Xtr, Ytr = Xtr[bank_size:], Ytr[bank_size:]
            else:
                raise ValueError(f'Invalid bank_size {bank_size}.')
            filename += f'_br={bank_ratio}'

        if fn:
            filename += f'_{fn}'

        weak_learner = RandomFilters(n_filters=n_filters, kernel_size=(ks, ks), init_filters=init_filters, filter_normalization=fn)

    elif wl == 'lcds' or 'local-convolution_decision-stump':
        weak_learner = LocalConvolution(weak_learner=MulticlassDecisionStump(), n_filters=n_filters, kernel_size=(ks, ks), init_filters=init_filters, locality=locality)
        filename += f'-nf={n_filters}-ks={ks}-loc={locality}-{init_filters}'
        kwargs['n_jobs'] = n_jobs
    logging.info(f'Weak learner: {type(weak_learner).__name__}')

    ### Callbacks
    ckpt = ModelCheckpoint(filename=filename+'-{round}.ckpt', dirname='./results', save_last=True)
    logger = CSVLogger(filename=filename+'-log.csv', dirname='./results/log')
    zero_risk = BreakOnZeroRiskCallback()

    callbacks = [ckpt,
                logger,
                zero_risk,
                ]

    logging.info(f'Filename: {filename}')

    ### Fitting the model
    if not resume:
        logging.info(f'Beginning fit with max_round_number={max_round} and patience={patience}.')
        qb = QuadBoostMHCR(weak_learner, encoder=encoder)
        qb.fit(Xtr, Ytr, max_round_number=max_round, patience=patience,
               X_val=Xts, Y_val=Yts,
               callbacks=callbacks,
               **kwargs)
    ### Or resume fitting a model
    else:
        logging.info(f'Resuming fit with max_round_number={max_round}.')
        qb = QuadBoostMHCR.load(f'results/{filename}-{resume}.ckpt')
        qb.resume_fit(Xtr, Ytr,
                      X_val=Xts, Y_val=Yts,
                      max_round_number=max_round,
                      **kwargs)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')
    main()
