from trainer.adavitTrainer import adavitTrainer
from trainer.base import BaseTrainer
from trainer.setup import args_parser, DebugArgs

# libgcc_s = ctypes.CDLL('libgcc_s.so.1')


if __name__ == '__main__':
    # args = args_parser()
    args= DebugArgs(model='deit_small_patch16_adaperturbed_vit', mode='train', keep_rate=0.7, batch_size=4, use_select_token=False, num_samples=100000)

    if 'vit' in args.model or 'deit' in args.model:
        trainer = adavitTrainer(args)
    else:
        trainer = BaseTrainer(args)

    if args.mode == 'visualize':
        trainer.visualize()
    elif args.mode == 'plot_attn_dist':
        trainer.plot_attn_dist(mode = ['dist', 'rank'], log_scale=True)
    elif args.mode == 'train':
        trainer.train()
        trainer.test(trainer.best_model_params)
        trainer.finish()
    elif args.mode == 'eval':
        trainer.test()
        trainer.finish()
    else:
        raise NotImplementedError(f'unrecognized mode: {args.mode}')


