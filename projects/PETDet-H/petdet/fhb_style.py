# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.dom.minidom as xml
from typing import List, Union

import mmcv
from mmengine.fileio import get

from mmdet.registry import DATASETS
from mmdet.datasets.xml_style import XMLDataset


@DATASETS.register_module()
class FHBDataset(XMLDataset):
    """XML dataset for detection.

    Args:
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = osp.join(self.sub_data_root, img_info['file_name'])
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, backend='cv2')
        height, width = img.shape[:2]
        del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        instances = []
        filehandle = open(img_info['xml_path'], 'rb')
        filecontent = filehandle.read()
        if filehandle is not None:
            filehandle.close()
        if filecontent is not None:
            filecontent.decode("utf-8", "ignore")

        dom = xml.parseString(filecontent)
        instances = []
        for obj in dom.getElementsByTagName('object'):
            instance = {}
            label = obj.getElementsByTagName('possibleresult')[
                0].getElementsByTagName('name')[0].firstChild.data
            points = obj.getElementsByTagName(
                'points')[0].getElementsByTagName('point')[:4]
            items = []
            for point in points:
                x, y = point.firstChild.data.split(',')
                items.append(x)
                items.append(y)

            if len(items) >= 8:
                qbox = [float(i) for i in items[:8]]
                x_coords = qbox[::2]
                y_coords = qbox[1::2]
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)
                instance['bbox'] = [xmin, ymin, xmax, ymax]
                instance['bbox_label'] = self.cat2label[label]
                instance['ignore_flag'] = 0
                instances.append(instance)
 
        data_info['instances'] = instances
        return data_info
