from trainer.adavitTrainer import adavitTrainer
from trainer.base import BaseTrainer
from trainer.helpers import args_parser, DebugArgs

# libgcc_s = ctypes.CDLL('libgcc_s.so.1')


if __name__ == '__main__':
    args = args_parser()
    # args= DebugArgs(model='deit_small_patch16_224_shrink_base', mode='train', keep_rate=1, batch_size=24, save_n_batch=5)

    if 'vit' in args.model or 'deit' in args.model:
        trainer = adavitTrainer(args)
    else:
        trainer = BaseTrainer(args)

    if args.mode == 'visualize':
        trainer.visualize()
    elif args.mode == 'plot_attn_dist':
        trainer.plot_attn_dist()
    elif args.mode == 'train':
        trainer.train()
        trainer.test(trainer.best_model_params)
        trainer.finish()
    elif args.mode == 'eval':
        trainer.test()
        trainer.finish()
    else:
        raise NotImplementedError(f'unrecognized mode: {args.mode}')


