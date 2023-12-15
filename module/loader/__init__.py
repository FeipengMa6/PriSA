from .dataset import basic_dataset,mosi_dataset,urfunny_dataset
def get_instance(name, cfg,mode):
    dataset = {'mosei_dataset': basic_dataset,'mosi':mosi_dataset,'urfunny':urfunny_dataset}[name]
    return dataset(cfg,mode)