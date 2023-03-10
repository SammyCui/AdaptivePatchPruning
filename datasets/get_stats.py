import os
import json

from datasets.data_utils import get_anno_stats


def get_bndbox_size_ratio(anno_root, output_path=None):
    stats = {}
    for cat in os.listdir(anno_root):
        bb_w, bb_h, bb_size_ratio, bb_x, bb_y, count = 0, 0, 0, 0, 0, 0
        for image_name in os.listdir(os.path.join(anno_root, cat)):
            annotation_path = os.path.join(anno_root, cat, image_name)
            x_min, y_min, x_max, y_max, img_w, img_h = get_anno_stats(annotation_path)
            bb_w += x_max - x_min
            bb_h += y_max - y_min
            cur_bb_size = (x_max - x_min) * (y_max - y_min)

            bb_size_ratio += cur_bb_size/(img_w * img_h)
            bb_x += x_min
            bb_y += y_min
            count += 1

        bb_w /= count
        bb_h /= count
        bb_size_ratio /= count
        bb_x /= count
        bb_y /= count

        stats[cat] = f"Size: {round(bb_w)} x {round(bb_h)}, Boundbox/IMG ratio: {round(bb_size_ratio,2)},  Loc[x_min, y_min] = [{round(bb_x)}, {round(bb_y)}]"

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
    print(stats)
    return stats


if __name__ == '__main__':
    get_bndbox_size_ratio('/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered/train/annotations',
                          './VOC_BB_stats.json')

