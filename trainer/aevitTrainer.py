from .base import BaseTrainer

class aevitTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def visualize(self):
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Visualize:'
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=device).reshape(3, 1, 1)
        std = torch.tensor(IMAGENET_DEFAULT_STD, device=device).reshape(3, 1, 1)

        # switch to evaluation mode
        model.eval()

        ii = 0
        for images, target in metric_logger.log_every(data_loader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            B = images.size(0)

            with torch.cuda.amp.autocast():
                output, idx = model(images, keep_rate, get_idx=True)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # denormalize
            images = images * std + mean

            idxs = get_real_idx(idx, fuse_token)
            for jj, idx in enumerate(idxs):
                masked_img = mask(images, patch_size=16, idx=idx)
                save_img_batch(masked_img, output_dir, file_name='img_{}' + f'_l{jj}.jpg',
                               start_idx=world_size * B * ii + rank * B)

            save_img_batch(images, output_dir, file_name='img_{}_a.jpg', start_idx=world_size * B * ii + rank * B)

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.synchronize_between_processes()
            ii += 1
            if world_size * B * ii >= n_visualization:
                break

        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}