import json
import sys
from mmdet.apis import DetInferencer


def write_head(f):
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


def write_obj(f,
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


def write_tail(f):
    tail = """    </objects>
    </annotation>
    """
    f.write(tail)


def hbox2qbox(boxes):
    """Convert horizontal boxes to quadrilateral boxes.

    Args:
        boxes (Tensor): horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Quadrilateral box tensor with shape of (..., 8).
    """
    x1, y1, x2, y2 = boxes
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def main():
    model = '/home/lwt/work/mmdetection/work_dirs/petdet-lite_fasternet_t1_fpn_300e_mar20_640/petdet-lite_fasternet_t1_fpn_300e_mar20_640.py'
    weights = '/home/lwt/work/mmdetection/work_dirs/petdet-lite_fasternet_t1_fpn_300e_mar20_640/epoch_300.pth'
    device = 'cuda:0'
    batch_size = 16
    pred_score_thr = 0.001

    input_json_path = sys.argv[-2]
    output_json_path = sys.argv[-1]

    with open(input_json_path, 'r') as file:
        input_json = file.read()
    input_path = json.loads(input_json)

    with open(output_json_path, 'r') as file:
        output_json = file.read()
    output_path = json.loads(output_json)

    inferencer = DetInferencer(model=model, weights=weights, device=device)
    result = inferencer(input_path,
                        batch_size=batch_size,
                        pred_score_thr=pred_score_thr)
    print('success')
    for xml_path, result in zip(output_path, result['predictions']):
        labels = result['labels']
        bboxes = result['bboxes']
        scores = result['scores']
        f = open(xml_path, 'w')
        write_head(f)
        for label, bbox, score in zip(labels, bboxes, scores):
            qbox = hbox2qbox(bbox)
            write_obj(f, str(label), qbox, str(score))
        write_tail(f)
        f.close()


if __name__ == "__main__":
    main()
