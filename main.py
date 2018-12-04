import torch
import logging

from quadboost import QuadBoostMHCR
from quadboost.label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from quadboost.weak_learner import *
from quadboost.callbacks import *
from quadboost.datasets import MNISTDataset
from quadboost.utils import parse, timed


@timed
@parse
def main(m=60_000, val=10_000, dataset='mnist', center=True, reduce=True, encodings='onehot', wl='rccridge', max_round=1000, patience=1000, resume=0, n_jobs=1, max_n_leaves=4, n_filters=10, fs=11, fsh=0, locality=4, init_filters='from_bank', bank_ratio=.05, fn='c', seed=42, nl='maxpool', maxpool=3, device='cpu', degrees=.0, scale=.0, shear=.0, margin=2, nt=1):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ### Data loading
    mnist = MNISTDataset.load(dataset+'.pkl')
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=center, reduce=reduce)
    idx = np.arange(m)
    if seed:
        np.random.shuffle(idx)
        val_idx = idx[:val]
        tr_idx = idx[val:]
    X_val, Y_val = Xtr[val_idx], Ytr[val_idx]
    Xtr, Ytr = Xtr[tr_idx], Ytr[tr_idx]
    logging.info(f'Loaded dataset: {dataset} (center: {center}, reduce: {reduce})')
    logging.info(f'Number of examples - train: {len(tr_idx)}, valid: {len(val_idx)}, test: {len(Xts)}')

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
    if wl in ['ds', 'decision-stump']:
        weak_learner = MulticlassDecisionStump()
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs

    elif wl in ['dt', 'decision-tree']:
        weak_learner = MulticlassDecisionTree(max_n_leaves=max_n_leaves)
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr)))
        kwargs['n_jobs'] = n_jobs
        filename += f'{max_n_leaves}'

    elif wl == 'ridge':
        weak_learner = WLThresholdedRidge(threshold=.5)

    elif wl.startswith('rcc') or wl.startswith('rlc'):
        if device.startswith('cuda'):
            Xtr = RandomConvolution.format_data(Xtr).to(device=device)
            X_val = RandomConvolution.format_data(X_val).to(device=device)
            Xts = RandomConvolution.format_data(Xts).to(device=device)

        filename += f'-nf={n_filters}-fs={fs}'
        if fsh: filename += f'_to_{fsh}'
        if wl.startswith('rlc'): filename += f'-loc={locality}'

        activation = None
        if 'maxpool' in nl:
            filename += f'-maxpool{maxpool}'
        if 'relu' in nl:
            filename += f'-relu'
            activation = torch.nn.functional.relu
        elif 'sigmoid' in nl:
            filename += f'-sigmoid'
            activation = torch.sigmoid

        filename += f'-{init_filters}'
        if degrees:
            filename += f'-deg={degrees}'
        if scale:
            filename += f'-scale={scale}'
            scale = (1-scale, 1/(1-scale))
        else:
            scale = None
        if shear:
            filename += f'-shear={shear}'
        else:
            shear = None

        filter_bank = None
        if init_filters == 'from_bank':
            if 0 < bank_ratio < 1:
                bank_size = int(m*bank_ratio)
                filter_bank = Xtr[:bank_size]
                Xtr, Ytr = Xtr[bank_size:], Ytr[bank_size:]
                logging.info(f'Bank size: {bank_size}')
            else:
                raise ValueError(f'Invalid bank_size {bank_size}.')
            filename += f'_br={bank_ratio}'
        elif init_filters == 'from_data':
            filter_bank = Xtr

        if fn:
            filename += f'_{fn}'

        f_proc = []
        if 'c' in fn:
            f_proc.append(center_weight)
        if 'n' in fn:
            f_proc.append(normalize_weight)
        if 'r' in fn:
            f_proc.append(reduce_weight)

        w_gen = WeightFromBankGenerator(filter_bank=filter_bank,
                                        filters_shape=(fs, fs),
                                        filters_shape_high=(fsh, fsh) if fsh else None,
                                        filter_processing=f_proc,
                                        margin=margin,
                                        degrees=degrees,
                                        scale=scale,
                                        shear=shear,
                                        )
        if wl.startswith('rcc'):
            filters = Filters(n_filters=n_filters,
                              weights_generator=w_gen,
                              activation=activation,
                              maxpool_shape=(nt, maxpool, maxpool))
        elif wl.startswith('rlc'):
            filters = LocalFilters(n_filters=n_filters,
                                   weights_generator=w_gen,
                                   locality=locality,
                                   maxpool_shape=(nt, maxpool, maxpool))
            if nt > 1:
                filename += f'-nt={nt}'
        if wl.endswith('ridge'):
            weak_learner = RandomConvolution(filters=filters, weak_learner=Ridge)
        if wl.endswith('ds'):
            weak_learner = RandomConvolution(filters=filters, weak_learner=MulticlassDecisionStump)
            kwargs['n_jobs'] = n_jobs

    else:
        raise ValueError(f'Invalid weak learner name: "{wl}".')

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
               X_val=X_val, Y_val=Y_val,
               callbacks=callbacks,
               **kwargs)
    ### Or resume fitting a model
    else:
        logging.info(f'Resuming fit with max_round_number={max_round}.')
        qb = QuadBoostMHCR.load(f'results/{filename}-{resume}.ckpt')
        qb.resume_fit(Xtr, Ytr,
                      X_val=X_val, Y_val=Y_val,
                      max_round_number=max_round,
                      **kwargs)

    print(f'Test accuracy on best model (round {qb.best_round}): {qb.evaluate(Xts, Yts):.3%}')
    print(f'Test accuracy on last model (round {len(qb.weak_predictors)}): {qb.evaluate(Xts, Yts, mode="last"):.3%}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, style='{', format='[{levelname}] {message}')
    main()
