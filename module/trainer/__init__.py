from . import prisa_trainer

def get_instance(name, cfg):
    trainer = {'prisa_trainer':prisa_trainer.prisa_trainer}[name]
    return trainer(cfg)