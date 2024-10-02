def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}_{P.adv_method}_{P.distance}_{P.augment_type}'

    if mode == 'adv_train':
        if P.consistency:
            if P.dummy:
                from .adv_consistency_dummy import train
                fname += f'_consistency_{P.lam}_temp_{P.T}_dummy'
            else:
                from .adv_consistency import train
                fname += f'_consistency_{P.lam}_temp_{P.T}'
        elif P.dummy:
            from .adv_train_dummy import train
            fname += f'_dummy'
        else:
            from .adv_train import train

    elif mode == 'adv_trades':
        if P.dummy:
            from .adv_trades_dummy import train
            fname += f'_dummy'
        else:
            from .adv_trades import train
            fname += f'_{P.beta}'

    elif mode == 'adv_mart':
        if P.dummy:
            from .adv_mart_dummy import train
            fname += f'_dummy'
        else:
            from .adv_mart import train
            fname += f'_{P.beta}'

    else:
        raise NotImplementedError()

    fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname
