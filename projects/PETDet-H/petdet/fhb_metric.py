# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.logging import MMLogger

from mmdet.evaluation.metrics import VOCMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class FHBMetric(VOCMetric):
    """DOTA evaluation metric.

    Note:  In addition to format the output results to JSON like CocoMetric,
    it can also generate the full image's results by merging patches' results.
    The premise is that you must use the tool provided by us to crop the DOTA
    large images, which can be found at: ``tools/data/dota/split``.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Only support
            'mAP' now. If is list, the first setting in the list will
             be used to evaluate metric.
        predict_box_type (str): Box type of model results. If the QuadriBoxes
            is used, you need to specify 'qbox'. Defaults to 'rbox'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format. Defaults to False.
        outfile_prefix (str, optional): The prefix of json/zip files. It
            includes the file path and the prefix of filename, e.g.,
            "a/b/prefix". If not specified, a temp file will be created.
            Defaults to None.
        merge_patches (bool): Generate the full image's results by merging
            patches' results.
        iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
            patches. Defaults to 0.1.
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'. Defaults to '11points'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'fhb'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 format_only: bool = True,
                 outfile_prefix: Optional[str] = None,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric

        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.use_07_metric = True if eval_mode == '11points' else False

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.

        You can submit it at:
        https://captain-whu.github.io/DOTA/evaluation.html

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the zip files. If the
                prefix is "somepath/xxx", the zip files will be named
                "somepath/xxx/xxx.zip".
        """
        collector = defaultdict(list)

        for idx, result in enumerate(results):
            img_path = result.get('img_path', idx)
            img_name = os.path.basename(img_path)
            basename, _ = os.path.splitext(img_name)

            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            label_dets = np.concatenate(
                [labels[:, np.newaxis], bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[basename].append(label_dets)

        basename_list, dets_list = [], []
        for basename, label_dets_list in collector.items():
            img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:  # noqa: E722
                        cls_dets = torch.from_numpy(dets[labels == i])
                    img_results.append(cls_dets.cpu().numpy())
            basename_list.append(basename)
            dets_list.append(img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'The outfile_prefix should be a non-exist path, '
                             f'but {outfile_prefix} is existing. '
                             f'Please delete it firstly.')
        os.makedirs(outfile_prefix)

        files = [
            osp.join(outfile_prefix, basename + '.xml')
            for basename in basename_list
        ]
        file_objs = [open(f, 'w') for f in files]
        for f, dets_per_cls in zip(file_objs, dets_list):
            self._write_head(f)
            for cls, dets in zip(self.dataset_meta['classes'], dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                hboxes, scores = torch.split(th_dets, (4, 1), dim=-1)
                qboxes = self.hbox2qbox(hboxes)
                for qbox, score in zip(qboxes, scores):
                    self._write_obj(f, cls, qbox, str(score.item()))
            self._write_tail(f)

        for f in file_objs:
            f.close()

        return outfile_prefix

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_path'] = data_sample['img_path']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape(
                        (-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(self.outfile_prefix)}')
            zip_path = self.merge_results(preds, self.outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            return super().compute_metrics(results)

    def _write_head(self,
                    f):
        head = """<?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <source>
            <id>placeholder_file_id</id>
            <filename>placeholder_filename</filename>
            <origin>placeholder_origin</origin>
            </source>
            <research>
                <version>1.0</version>
                <provider>placeholder_affiliation</provider>
                <author>placeholder_authorname</author>
                <!--参赛课题 -->
                <pluginname>placeholder_direction</pluginname>
                <pluginclass>placeholder_suject</pluginclass>
                <testperson>Airvic</testperson>
                <time>2023-10</time>
            </research>
            <!--存放目标检测信息-->
            <objects>
        """
        f.write(head)

    def _write_obj(self,
                   f,
                   cls: str,
                   bbox,
                   conf: float):
        obj_str = """        <object>
                    <coordinate>pixel</coordinate>
                    <type>rectangle</type>
                    <description>None</description>
                    <possibleresult>
                        <name>palceholder_cls</name>                
                        <probability>palceholder_prob</probability>
                    </possibleresult>
                    <!--检测框坐标，首尾闭合的矩形，起始点无要求-->
                    <points>  
                        <point>palceholder_coord0</point>
                        <point>palceholder_coord1</point>
                        <point>palceholder_coord2</point>
                        <point>palceholder_coord3</point>
                        <point>palceholder_coord0</point>
                    </points>
                </object>
        """
        obj_xml = obj_str.replace("palceholder_cls", cls)
        obj_xml = obj_xml.replace("palceholder_prob", conf)
        obj_xml = obj_xml.replace(
            "palceholder_coord0", f'{bbox[0]:.2f}'+", "+f'{bbox[1]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord1", f'{bbox[2]:.2f}'+", "+f'{bbox[3]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord2", f'{bbox[4]:.2f}'+", "+f'{bbox[5]:.2f}')
        obj_xml = obj_xml.replace(
            "palceholder_coord3", f'{bbox[6]:.2f}'+", "+f'{bbox[7]:.2f}')
        f.write(obj_xml)

    def _write_tail(self,
                    f):
        tail = """    </objects>
        </annotation>
        """
        f.write(tail)

    def hbox2qbox(self, boxes):
        """Convert horizontal boxes to quadrilateral boxes.

        Args:
            boxes (Tensor): horizontal box tensor with shape of (..., 4).

        Returns:
            Tensor: Quadrilateral box tensor with shape of (..., 8).
        """
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
        return torch.cat([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)
