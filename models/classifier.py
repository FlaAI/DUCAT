from models.wide_resnet import wide_resnet_28_10
from models.resnet import resnet18


def get_classifier(P, n_classes=10):
    if P.dummy:
        n_classes *= 2

    if P.model == 'resnet18':
        if P.dataset == 'tinyimagenet':
            classifier = resnet18(num_classes=n_classes, stride=2)
        else:
            classifier = resnet18(num_classes=n_classes)
    elif P.model == 'wrn2810':
        if P.dataset == 'tinyimagenet':
            classifier = wide_resnet_28_10(num_classes=n_classes, stride=2)
        else:
            classifier = wide_resnet_28_10(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier
