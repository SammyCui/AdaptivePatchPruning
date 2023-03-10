import xmltodict

def get_anno_stats(anno_path: str):
    cat = anno_path.split('/')[-2]
    with open(anno_path, "r") as xml_obj:
        # coverting the xml data to Python dictionary
        my_dict = xmltodict.parse(xml_obj.read())
        # closing the file
        xml_obj.close()

    obj = my_dict['annotation']['object']
    img_w, img_h = int(my_dict['annotation']['size']['width']), int(my_dict['annotation']['size']['height'])
    if isinstance(obj, list):
        h_bb_max, w_bb_max = 0, 0
        for object in obj:
            if object['name'] == cat:
                cur_x_min = int(object['bndbox']['xmin'])
                cur_y_min = int(object['bndbox']['ymin'])
                cur_x_max = int(object['bndbox']['xmax'])
                cur_y_max = int(object['bndbox']['ymax'])
                if (cur_y_max - cur_y_min) * (cur_x_max - cur_x_min) > h_bb_max * w_bb_max:
                    x_min, y_min, x_max, y_max = cur_x_min, cur_y_min, cur_x_max, cur_y_max
                    h_bb_max, w_bb_max = y_max - y_min, x_max - x_min
        assert (h_bb_max != 0) or (w_bb_max != 0), f"Invalid bndbox annotation: {anno_path}"
    else:
        x_min = int(obj['bndbox']['xmin'])
        y_min = int(obj['bndbox']['ymin'])
        x_max = int(obj['bndbox']['xmax'])
        y_max = int(obj['bndbox']['ymax'])

    return x_min, y_min, x_max, y_max, img_w, img_h