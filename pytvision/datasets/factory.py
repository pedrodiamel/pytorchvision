import os
from enum import Enum

import numpy as np
from torchvision import datasets

from . import afew, affect, cars196, celeba, cub2011, fer, ferfolder, ferp, imaterialist, stanford_online_products


def create_folder(pathname, name):
    # create path name dir
    pathname = os.path.join(pathname, name)
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    return pathname


class FactoryDataset(object):

    # Training subsets
    class Subsets(Enum):
        TRAIN = 1
        VAL = 2
        TEST = 3

        def to_str(self):
            return self.name.lower()

    # DATASETS NAMES
    class Datasets(Enum):

        MNIST = 1
        FASHION = 2
        EMNIST = 3
        CIFAR10 = 4
        CIFAR100 = 5
        STL10 = 6
        SVHN = 7
        IMATERIALIST = 8
        FERP = 9
        CK = 10
        JAFFE = 11
        BU3DFE = 12
        AFEW = 13
        CELBA = 14
        CUB2011 = 16
        CARS196 = 17
        STANFORD_ONLINE_PRODUCTS = 18
        CUB2011METRIC = 19
        CARS196METRIC = 20
        CKP = 21
        AFFECTNET = 22

        def to_str(self):
            return self.name.lower()

        def to_dataset(self, name):
            return self.str_to_dataset[name]

    # Create str to dataset
    # Exemplo
    # {
    #     Datasets.MNIST.to_str(): Datasets.MNIST
    # }
    str_to_dataset = {e.to_str(): e for e in Datasets.__members__.values()}

    @classmethod
    def factory(
        self,
        pathname: str,
        name: Datasets,
        subset: Subsets = Subsets.TRAIN,
        idenselect=[],
        download=False,
        transform=None,
    ):
        """
        Methodo factory to create dataset
        Arguments:
            pathname (str): path of the datasets
            name (Datasets): enum datasets names support
            subset (Subsets): TRAIN/VAL/TEST
        """

        pathname = os.path.expanduser(pathname)
        btrain = subset == self.Subsets.TRAIN

        # create folder
        pathname = create_folder(pathname, name.to_str())

        data = None
        # pythorch vision dataset suported
        if name == self.Datasets.MNIST:
            data = datasets.MNIST(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.FASHION:
            data = datasets.FashionMNIST(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.EMNIST:
            data = datasets.EMNIST(pathname, split="byclass", train=btrain, transform=transform, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.CIFAR10:
            data = datasets.CIFAR10(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.CIFAR100:
            data = datasets.CIFAR100(pathname, train=btrain, transform=transform, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.STL10:
            split = "train" if (btrain) else "test"
            data = datasets.STL10(pathname, split=split, transform=transform, download=download)

        elif name == self.Datasets.SVHN:
            split = "train" if (btrain) else "test"
            data = datasets.SVHN(pathname, split=split, transform=transform, download=download)
            data.classes = np.unique(data.labels)

        # internet dataset

        elif name == self.Datasets.CUB2011:
            data = cub2011.CUB2011(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.CARS196:
            data = cars196.Cars196(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.STANFORD_ONLINE_PRODUCTS:
            data = stanford_online_products.StanfordOnlineProducts(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)
            data.btrain = btrain

        # kaggle dataset
        elif name == self.Datasets.IMATERIALIST:
            data = imaterialist.IMaterialistDatset(pathname, subset.to_str(), "jpg")

        # fer recognition datasets

        elif name == self.Datasets.FERP:
            if subset == self.Subsets.TRAIN:
                subfolder = ferp.train
            elif subset == self.Subsets.VAL:
                subfolder = ferp.valid
            elif subset == self.Subsets.TEST:
                subfolder = ferp.test
            else:
                raise ValueError("Dataset {} not suport the subset: ".format(subset.to_str()))
            data = ferp.FERPDataset(pathname, subfolder, download=download)

        elif name == self.Datasets.CK:
            idenselect = np.arange(20) + 0
            data = fer.FERClassicDataset(pathname, "ck", idenselect=idenselect, train=btrain)

        elif name == self.Datasets.CKP:
            data = fer.FERClassicDataset(pathname, "ckp", idenselect=idenselect, train=btrain)

        elif name == self.Datasets.JAFFE:
            data = fer.FERClassicDataset(pathname, "jaffe", idenselect=idenselect, train=btrain)

        elif name == self.Datasets.BU3DFE:
            # idenselect = np.array([0,1,2,3,4,5,6,7,8,9]) + 0
            data = fer.FERClassicDataset(pathname, "bu3dfe", idenselect=idenselect, train=btrain)

        elif name == self.Datasets.AFEW:
            data = afew.Afew(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.CELBA:
            data = celeba.CelebaDataset(pathname, train=btrain, download=download)

        elif name == self.Datasets.AFFECTNET:
            data = affect.create_affect(path=pathname, train=btrain)

        # metric learning dataset

        elif name == self.Datasets.CUB2011METRIC:
            data = cub2011.CUB2011MetricLearning(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)

        elif name == self.Datasets.CARS196METRIC:
            data = cars196.Cars196MetricLearning(pathname, train=btrain, download=download)
            data.labels = np.array(data.targets)

        else:
            raise ValueError("Dataset {} not suport".format(name.to_str()))

        data.btrain = btrain
        return data
