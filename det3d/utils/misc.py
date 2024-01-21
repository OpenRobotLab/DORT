def modify_file_client_args(cfg):
    file_client_args = dict()
    file_client_args['backend'] = 'petrel'
    file_client_args['path_mapping'] = dict()
    source = cfg.data_root
    if source.startswith('./'):
        source = source[2:]
    target = source.replace('data/', 's3://openmmlab/datasets/detection3d/')
    source1 = './' + source
    file_client_args['path_mapping'][source1] = target
    file_client_args['path_mapping'][source] = target

    for idx, item in enumerate(cfg.train_pipeline):
        if item['type'] == 'LoadMultiViewImageFromFiles' or \
            item['type'] == 'CustomLoadMultiViewImageFromFiles' or \
            item['type'] == 'LoadImageFromFileMono3D' or \
            item['type'] == 'LoadImageFromFile' or \
            item['type'] == 'BevDetLoadMultiViewImageFromFiles' or \
            item['type'] == 'LoadPointsFromFile' or \
            item['type'] == 'BevDetLoadPointsFromFile':

                item['file_client_args'] = file_client_args
                # break
                cfg.train_pipeline[idx] = item

    for idx, item in enumerate(cfg.test_pipeline):
        if item['type'] == 'LoadMultiViewImageFromFiles' or \
            item['type'] == 'CustomLoadMultiViewImageFromFiles' or \
            item['type'] == 'LoadImageFromFileMono3D' or \
            item['type'] == 'LoadImageFromFile' or \
            item['type'] == 'BevDetLoadMultiViewImageFromFiles' or \
            item['type'] == 'LoadPointsFromFile' or \
            item['type'] == 'BevDetLoadPointsFromFile':
            
                item['file_client_args'] = file_client_args
                # break
                cfg.test_pipeline[idx] = item

    if 'pipeline' in cfg.data.train:
        cfg.data.train['pipeline'] = cfg.train_pipeline
    if 'dataset' in cfg.data.train and 'pipeline' in cfg.data.train.dataset:
        cfg.data.train.dataset['pipeline'] = cfg.train_pipeline
    if 'datasets' in cfg.data.train:
        datasets = []
        for temp in cfg.data.train.datasets:
            if 'pipeline' in temp:
                temp['pipeline'] = cfg.train_pipeline
            if 'dataset' in temp and 'pipeline' in temp.dataset:
                temp['dataset']['pipeline'] = cfg.train_pipeline
            datasets.append(temp)
        cfg.data.train.datasets = datasets


    if 'pipeline' in cfg.data.val:
        cfg.data.val['pipeline'] = cfg.test_pipeline
    if 'dataset' in cfg.data.val and 'pipeline' in cfg.data.val.dataset:
        cfg.data.val.dataset['pipeline'] = cfg.test_pipeline
    if 'datasets' in cfg.data.val and 'pipeline' in cfg.data.val.datasets[0]:
        cfg.data.val.datasets[0]['pipeline'] = cfg.test_pipeline
    if 'datasets' in cfg.data.val and 'pipeline' in cfg.data.val.datasets[1]:
        cfg.data.val.datasets[1]['pipeline'] = cfg.test_pipeline



    if 'pipeline' in cfg.data.test:
        cfg.data.test['pipeline'] = cfg.test_pipeline
    if 'dataset' in cfg.data.test and 'pipeline' in cfg.data.test.dataset:
        cfg.data.test.dataset['pipeline'] = cfg.test_pipeline

    '''
    log_hooks = []
    for hook in cfg.log_config.hooks:
        if hook['type'] != 'WandbLoggerHook':
            log_hooks.append(hook)
    cfg.log_config['hooks'] = log_hooks
    '''
    return cfg