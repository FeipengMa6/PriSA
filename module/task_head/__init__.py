from .task_head import multi_classification_head, regression_head,classification_head
def get_instance(name,cfg,modality):
    task = {'regression': regression_head,
            'classification':regression_head}[name]
    return task(cfg,modality)