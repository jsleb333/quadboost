import torch
import logging

from quadboost.label_encoder import OneHotEncoder
from quadboost.weak_learner import *
from quadboost.datasets import MNISTDataset, CIFAR10Dataset
from quadboost.utils import parse, timed

from py2tex import Document, Plot

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



seed = 101
torch.manual_seed(seed)
np.random.seed(seed)

mnist = MNISTDataset.load()
(Xtr, Ytr), (Xts, Yts) = mnist.get_train_test(center=True, reduce=True)
Xtr = torch.unsqueeze(torch.from_numpy(Xtr), dim=1)
Xts = torch.unsqueeze(torch.from_numpy(Xts), dim=1)
# encoder = OneHotEncoder(Ytr)

print('CUDA')
Xtr = Xtr.to(device='cuda:0')
Xts = Xts.to(device='cuda:0')

@timed
def n_features_plot(nf,
                    m=1000,
                    bank=3000,
                    nt=10,
                    rotation=0,
                    scale=0,
                    shear=0
                    ):
    seed = 101
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'n filters = {nf}, n transform = {nt}')
    filter_gen = WeightFromBankGenerator(filter_bank=Xtr[-bank:],
                                        filters_shape=(11,11),
                                        # filters_shape=(5,5),
                                        # filters_shape_high=(16,16),
                                        margin=2,
                                        filter_processing=[center_weight],
                                        rotation=rotation,
                                        scale=scale,
                                        shear=shear,
                                        n_transforms=nt,
                                        )
    filters = LocalFilters(n_filters=nf,
                        maxpool_shape=(-1,-1,-1),
                        weights_generator=filter_gen,
                        locality=3,
                        )
    weak_learner = Ridge
    wl = SVC(gamma='scale')

    print('Computing random features')
    X = RandomConvolution.format_data(Xtr[:m])
    random_features = filters.apply(X)
    X_test = RandomConvolution.format_data(Xts[:m])
    random_features_test = filters.apply(X_test)

    # wl = WLRidge(encoder=encoder)
    print('Starting fit')
    wl.fit(random_features, Ytr[:m])
    tr_acc = accuracy_score(wl.predict(random_features), Ytr[:m])

    ts_acc = accuracy_score(wl.predict(random_features_test), Yts[:m])
    print('Train acc', tr_acc)
    print('Test acc', ts_acc)

    return tr_acc, ts_acc


if __name__ == '__main__':
    nt = 40
    # scale = (1,1)
    # shear = 0
    # rotation = 0
    scale = .1
    shear = 10
    rotation = 15

    # doc_name = f'n_features_plot_n_transforms={nt}'
    # doc = Document(doc_name)
    # plot = doc.new(Plot(plot_name=doc_name))

    # nfs = list(np.linspace(10,90,9,dtype=int)) + list(np.linspace(100,700,7,dtype=int)) + [784,800,900,1000]
    # # nfs = [10,20,30]
    # tr_accs, ts_accs = [], []
    # for nf in nfs:
    #     tr_acc, ts_acc = n_features_plot(nf, m=57000, nt=nt, rotation=rotation, scale=scale, shear=shear)
    #     tr_accs.append(tr_acc)
    #     ts_accs.append(ts_acc)

    n_features_plot(784, m=60_000)

    # plot.add_plot(nfs, tr_accs)
    # plot.add_plot(nfs, ts_accs)

    # plot.y_min = 0
    # plot.y_max = 1

    # doc.build()

