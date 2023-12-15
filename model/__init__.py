from .prisa_model import prisa_model
def get_instance(name, cfg):
    model = {'prisa_model':prisa_model}[name]
    return model(cfg)
