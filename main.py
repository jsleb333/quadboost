from utils import parse
from quadboost import QuadBoostMHCR
from callbacks import ModelCheckpoint, CSVLogger
from label_encoder import LabelEncoder, OneHotEncoder, AllPairsEncoder
from mnist_dataset import MNISTDataset
from weak_learner import WLRidge, WLThresholdedRidge, MulticlassDecisionStump
import logging


@parse
def main(m=60_000, dataset='haar_mnist', encodings='ideal_mnist', wl='ds', n_jobs=1, max_round=400, patience=10):
    ### Data loading
    mnist = MNISTDataset.load(dataset+'.pkl')
    (Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=False, reduce=False)

    ### Choice of encoder
    if encodings == 'onehot':
        encoder = OneHotEncoder(Ytr)
    elif encodings == 'allpairs':
        encoder = AllPairsEncoder(Ytr)
    else:
        encoder = LabelEncoder.load_encodings(encodings)
        if all(label.isdigit() for label in encoder.labels_encoding):
            encoder = LabelEncoder({int(label):encoding for label, encoding in encoder.labels_encoding.items()})

    ### Choice of weak learner
    if wl == 'ds' or 'decision-stump':
        weak_learner = MulticlassDecisionStump()
        kwargs = dict(zip(('sorted_X', 'sorted_X_idx'), weak_learner.sort_data(Xtr[:m])))
        kwargs['n_jobs'] = n_jobs
    elif wl == 'ridge':
        weak_learner = WLThresholdedRidge(threshold=.5)
        kwargs = {}

    ### Callbacks
    filename = f'{dataset}_{encodings}_{wl}_'
    ckpt = ModelCheckpoint(filename=filename+'{round}.ckpt', dirname='./results', save_last=True)
    logger = CSVLogger(filename=filename+'log.csv', dirname='./results/log')
    callbacks = [ckpt,
                logger,
                ]

    ### Fitting the model
    qb = QuadBoostMHCR(weak_learner, encoder=encoder)
    qb.fit(Xtr[:m], Ytr[:m], max_round_number=max_round, patience=patience,
           X_val=Xts, Y_val=Yts,
           callbacks=callbacks,
           **kwargs)
    ### Or resume fitting a model
    # qb = QuadBoostMHCR.load('results/test2.ckpt')
    # qb.resume_fit(Xtr[:m], Ytr[:m],
    #               X_val=Xts, Y_val=Yts,
    #               n_jobs=1, sorted_X=sorted_X, sorted_X_idx=sorted_X_idx)

if __name__ == '__main__':
    main()