import time

from tensorboardX import SummaryWriter
import json
import os
from enum import Enum
import pandas as pd


class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


class Logger:
    def __init__(self,
                 args,
                 **kwargs       # kwargs for tensorboard SummaryWriter
                 ):

        self.result_dir = args.result_dir
        self.tb_logger = SummaryWriter(
            logdir=os.path.join(self.result_dir, 'tflogger'),
            **kwargs
        )

        self.args = args
        self.scalars = pd.DataFrame()

    def add_scalar(self, key, value, counter):
        self.scalars.loc[counter, key] = value
        self.tb_logger.add_scalar(key, value, counter)

    def add_scalars(self, main_key, key_val_dict, counter):
        self.scalars.loc[counter] = key_val_dict
        self.tb_logger.add_scalars(main_tag=main_key,
                                   tag_scalar_dict=key_val_dict,
                                   global_step=counter)

    def save_logger(self):
        # save scalars
        self.scalars.to_pickle(os.path.join(self.result_dir, 'scalars.pkl'))

        # save configurations
        config_filepath = os.path.join(self.result_dir, 'configs.json')
        with open(config_filepath, "w") as fd:
            json.dump(vars(self.args), fd, indent=2, sort_keys=True, cls=ConfigEncoder)

    def close(self):
        self.tb_logger.close()


class Timer:

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)
