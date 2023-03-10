import scipy
import scipy.stats
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class StatsMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.record = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.record = []

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.record.append(val)

    def compute_ci(self, confidence=0.95):
        """

        :return: confidence interval of current record
        """

        a = 1.0 * np.array(self.record)
        n = len(a)
        se = scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2, n-1)
        return h




