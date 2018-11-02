from utils import parse, timed
from datetime import datetime
from quadboost import QuadBoostMHCR
from callbacks import ModelCheckpoint, CSVLogger, BreakOnZeroRiskCallback
from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from weak_learner import WLRidge, WLThresholdedRidge, MulticlassDecisionStump, MulticlassDecisionTree, RandomFilters
import logging


@timed
@parse
def main(m=60_000, dataset='haar_mnist', encodings='onehot', wl='dt', n_jobs=1, max_n_leaves=4, max_round=1000, patience=1000, resume=0, n_filters=3, kernel_size=5, init_filters='from_data', center=False, reduce=False):
    ### Data loading
    mnist = MNISTDataset.load(dataset+'.pkl')
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=center, reduce=reduce)
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
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr[:m])))
        kwargs['n_jobs'] = n_jobs
    elif wl == 'dt' or wl == 'decision-tree':
        weak_learner = MulticlassDecisionTree(max_n_leaves=max_n_leaves)
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr[:m])))
        kwargs['n_jobs'] = n_jobs
        filename += f'{max_n_leaves}'
    elif wl == 'ridge':
        weak_learner = WLThresholdedRidge(threshold=.5)
    elif wl == 'rf' or wl == 'random_filters':
        weak_learner = RandomFilters(n_filters=n_filters, kernel_size=(kernel_size, kernel_size), init_filters=init_filters)
        filename += f'-nf={n_filters}-ks={kernel_size}-{init_filters}'
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
        qb.fit(Xtr[:m], Ytr[:m], max_round_number=max_round, patience=patience,
               X_val=Xts, Y_val=Yts,
               callbacks=callbacks,
               **kwargs)
    ### Or resume fitting a model
    else:
        logging.info(f'Resuming fit with max_round_number={max_round}.')
        qb = QuadBoostMHCR.load(f'results/{filename}{resume}.ckpt')
        qb.resume_fit(Xtr[:m], Ytr[:m],
                      X_val=Xts, Y_val=Yts,
                      max_round_number=max_round,
                      **kwargs)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')
    main()
