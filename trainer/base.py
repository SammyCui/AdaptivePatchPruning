import abc
import json
import torch
import os
import torch.nn.functional as F
from timm.data import Mixup
import datetime
from utils.logger import Logger
from utils.meters import AverageMeter, StatsMeter
from utils.metric import accuracy
from timeit import default_timer as timer
from copy import deepcopy
from trainer.setup import get_model_optimizer, get_dataloaders
from fvcore.nn import FlopCountAnalysis


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.logger = Logger(args)
        self.args = args
        self.train_step = 0
        self.train_epoch = args.start_epoch
        self.start_epoch = args.start_epoch
        self.max_epoch = args.max_epoch
        self.device = args.device

        self.train_time, self.forward_tm, self.backward_tm, self.optimize_tm = (deepcopy(AverageMeter()) for _ in range(4))

        self.result_log = {'max_val_acc@1': 0,
                           'max_val_acc@1_epoch': 0}
        self.train_dataloader, self.val_dataloader, self.test_dataloader, num_classes = get_dataloaders(args)
        args.num_classes = num_classes
        self.model, self.criterion, self.optimizer, self.lr_scheduler = get_model_optimizer(args)
        self.model.to(self.device)

        self.img_per_sec = None
        self.best_model_params = None
        self.gflop_total_per_img = None

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
        self.mixup_fn = mixup_fn

        print(json.dumps(vars(args), indent=2))

    def _train_one_epoch(self, **kwargs):

        self.model.train()

        train_loss, train_acc_1, train_acc_3, train_acc_5 = (deepcopy(AverageMeter()) for _ in range(4))
        for data, labels in self.train_dataloader:

            self.train_step += 1
            data, labels = data.to(self.device), labels.to(self.device)

            if self.mixup_fn:
                data, labels = self.mixup_fn(data, labels)

            with torch.cuda.amp.autocast():
                forward_t0 = timer()
                outputs = self.model(data, **kwargs)
                forward_t1 = timer()

                loss = self.criterion(outputs, labels)

            acc_1, acc_3, acc_5 = accuracy(outputs, labels, topk=(1, 3, 5))

            self.optimizer.zero_grad()
            backward_t0 = timer()
            loss.backward()
            backward_t1 = timer()

            optimizer_t0 = timer()
            self.optimizer.step()
            optimizer_t1 = timer()

            train_loss.update(loss.item())
            train_acc_1.update(acc_1[0].item())
            train_acc_3.update(acc_3[0].item())
            train_acc_5.update(acc_5[0].item())
            self.backward_tm.update(backward_t1 - backward_t0)
            self.forward_tm.update(forward_t1 - forward_t0)
            self.optimize_tm.update(optimizer_t1 - optimizer_t0)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        val_acc_1, val_acc_3, val_acc_5, val_loss = self.validate(**kwargs)

        self.logging(train_loss=train_loss.avg, train_acc=train_acc_1.avg,
                     val_loss=val_loss, val_acc=val_acc_1)
        self.train_epoch += 1

    def train(self):
        print('==> Training Start')

        for epoch in range(self.start_epoch, self.max_epoch+1):
            epoch_t0 = timer()
            self._train_one_epoch()
            self.train_time.update(timer() - epoch_t0)

    def _validate(self, **kwargs):
        self.model.eval()
        val_loss, val_acc_1, val_acc_3, val_acc_5 = (deepcopy(StatsMeter()) for _ in range(4))
        with torch.no_grad():
            for data, labels in self.val_dataloader:

                data, labels = data.to(self.device), labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(data, **kwargs)

                    loss = F.cross_entropy(outputs, labels)
                acc_1, acc_3, acc_5 = accuracy(outputs, labels, topk=(1,3,5))

                val_loss.update(loss.item())
                val_acc_1.update(acc_1[0].item())
                val_acc_3.update(acc_3[0].item())
                val_acc_5.update(acc_5[0].item())


        return val_acc_1.avg, val_acc_3.avg, val_acc_5.avg, val_loss.avg

    def validate(self, **kwargs):
        if self.train_epoch % self.args.val_interval == 0:
            val_acc_1, val_acc_3, val_acc_5, val_loss = self._validate(**kwargs)

            if val_acc_1 >= self.result_log['max_val_acc@1']:
                self.result_log['max_val_acc@1'] = val_acc_1
                self.result_log['max_val_acc@1_epoch'] = self.train_epoch
                self.result_log['val_acc@3_@maxacc@1'] = val_acc_3
                self.result_log['val_acc@5_@maxacc@1'] = val_acc_5
                self.best_model_params = self.model.state_dict()
                if self.args.save:
                    self.save_model('checkpoint')

            return val_acc_1, val_acc_3, val_acc_5, val_loss

    def save_model(self, name):
        assert self.model is not None, "No models to be saved."
        checkpoint = {'models': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if self.lr_scheduler:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.args.result_dir, name + '.pt'))

    def logging(self,
                train_loss,
                train_acc,
                val_loss,
                val_acc):
        assert self.optimizer is not None, "Has not initialize optimizer yet."

        if self.train_epoch % self.args.val_interval == 0:
            log_str = 'epoch {}/{}, **Train** loss={:.4f} acc={:.4f} | ' \
                  '**Val** loss={:.4f} acc@1={:.4f}'.format(self.train_epoch,
                                                      self.max_epoch,
                                                      train_loss, train_acc,
                                                      val_loss, val_acc)
            for param_group in self.optimizer.param_groups:
                log_str += " lr{}={:.4g}".format(
                    "-" + param_group['name'] if 'name' in param_group else "",
                    param_group['lr']
                )
            print(log_str)

            self.logger.add_scalar('train_loss', train_loss, self.train_epoch)
            self.logger.add_scalar('train_acc', train_acc, self.train_epoch)
            self.logger.add_scalar('val_loss', val_loss, self.train_epoch)
            self.logger.add_scalar('val_acc@1', val_acc, self.train_epoch)

    def test(self, param=None):
        print('==> Testing start')
        if param:
            self.model.load_state_dict(param)
        self.model.eval()
        t0 = timer()
        test_loss, test_acc_1, test_acc_3, test_acc_5 = (deepcopy(StatsMeter()) for _ in range(4))
        with torch.no_grad():
            for data, labels in self.test_dataloader:

                data, labels = data.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = F.cross_entropy(outputs, labels)

                acc_1, acc_3, acc_5 = accuracy(outputs, labels, topk=(1,3,5))

                test_loss.update(loss.item())
                test_acc_1.update(acc_1[0].item())
                test_acc_3.update(acc_3[0].item())
                test_acc_5.update(acc_5[0].item())

                if not self.gflop_total_per_img:
                    flop_total = FlopCountAnalysis(self.model, data).total()
                    self.gflop_total_per_img = flop_total // labels.shape[0] / 1e9

        self.img_per_sec = (len(self.test_dataloader) * self.args.batch_size) / (timer() - t0)

        self.result_log['test_acc@1'] = test_acc_1.avg
        self.result_log['test_acc@3'] = test_acc_3.avg
        self.result_log['test_acc@5'] = test_acc_5.avg
        self.result_log['test_loss'] = test_loss.avg

    def finish(self):
        self.logger.save_logger()
        print("==>", 'Training Statistics')

        for k, v in self.result_log.items():
            print(k, ': ', '{:.3f}'.format(v))

        if self.args.mode == 'train':
            print(
                'forward_timer  (avg): {:.2f} sec  \n' \
                'backward_timer (avg): {:.2f} sec, \n' \
                'optim_timer (avg): {:.2f} sec \n' \
                'epoch_timer (avg): {:.5f} hrs \n' \
                'total time to converge: {:.2f} hrs \n' \
                'throughput per sec: {:.2f}\n' \
                'total gflops per image: {:.2f} \n' \
                'finished at {} \n'
                    .format(
                        self.forward_tm.avg, self.backward_tm.avg,
                        self.optimize_tm.avg, self.train_time.avg / 3600,
                        self.train_time.sum / 3600,
                        self.img_per_sec,
                        self.gflop_total_per_img,
                        datetime.datetime.now()
                    )
            )
            if self.args.write_to_collections:
                with open(self.args.write_to_collections, 'a') as f:
                    f.write('=' * 50 + '\n')
                    f.write(self.args.run_name + ': \n')
                    f.write('\t Best epoch {}, best val acc={:.4f}\n'.format(
                        self.result_log['max_val_acc@1_epoch'],
                        self.result_log['max_val_acc@1']))
                    f.write('\t Test acc@1={:.4f} acc@3={:.4f} acc@5={:.4f}\n'.format(
                        self.result_log['test_acc@1'], self.result_log['test_acc@3'], self.result_log['test_acc@5']))
                    f.write('\t total time to converge: {:.3f} hrs, per epoch: {:.5f} hrs \n'
                            .format(self.train_time.sum / 3600, self.train_time.avg / 3600))
                    f.write('\t throughput per sec: {:.2f} \n'.format(self.img_per_sec))
                    f.write('\t gflops per image: {:.2f} \n'.format(self.gflop_total_per_img))
                    f.write(f'\t finished at: {datetime.datetime.now()} \n')
                    f.write('=' * 50 + '\n')

        else:
            print('throughput per sec: {:.2f} \n' \
                  'gflops per image: {:.2f}'.format(self.img_per_sec, self.gflop_total_per_img))

            with open(os.path.join(self.args.result_dir, 'results.txt'), 'w') as f:
                f.write('=' * 50 + '\n')
                f.write(self.args.run_name + ': \n')
                f.write('\t Test acc@1={:.4f} acc@3={:.4f} acc@5={:.4f}\n'.format(
                    self.result_log['test_acc@1'], self.result_log['test_acc@3'], self.result_log['test_acc@5']))
                f.write('\t throughput per sec: {:.2f} \n'.format(self.img_per_sec))
                f.write('\t Total model gflops per image: {:.2f} \n'.format(self.gflop_total_per_img))
                f.write(f'\t finished at: {datetime.datetime.now()} \n')
                f.write('=' * 50 + '\n')

        with open(os.path.join(self.args.result_dir, 'model_arch.txt'), 'w') as f:
            f.write(str(self.model))

        self.logger.close()

    def __str__(self):
        return "{}({}). \n Args: {}".format(
            self.__class__.__name__,
            self.model.__class__.__name__,
            json.dumps(vars(self.args), indent=2)
        )


