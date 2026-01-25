
"""Convert dataset from PASCAL VOC XML format to YOLO format."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_bbox_to_yolo(
    size: tuple[float, float], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert bounding box from absolute coordinates to relative coordinates.

    :param size: Tuple of (width, height) of the image.
    :param box: Tuple of (xmin, ymin, xmax, ymax) for the bounding box.
    :return: Tuple of (x_center, y_center, width, height) in relative
        coordinates.
    """
    scale_width = 1.0 / size[0]
    scale_height = 1.0 / size[1]

    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]

    rel_center_x = center_x * scale_width
    rel_center_y = center_y * scale_height
    rel_width = box_width * scale_width
    rel_height = box_height * scale_height

    return (rel_center_x, rel_center_y, rel_width, rel_height)


def xml_to_txt(input_xml: str, output_txt: str, class_mapping: dict[str, int]):
    """Parse an XML file and write to a .txt file in YOLO format.

    :param input_xml: Path to the input XML file.
    :param output_txt: Path to the output .txt file.
    :param class_mapping: Dictionary mapping class names to class.
    """
    tree = ET.parse(input_xml)
    root = tree.getroot()
    width = int(root.find(".//size/width").text)
    height = int(root.find(".//size/height").text)


    with open(output_txt, "w", encoding="utf-8") as txt_file:
        for obj in root.iter("object"):
            cell_name = obj.find("name").text
            class_id = class_mapping.get(cell_name, -1)

            if class_id == -1:
                continue

            xmlbox = obj.find("bndbox")
            box = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )
            bbox = convert_bbox_to_yolo((width, height), box)
            txt_file.write(f"{class_id} {' '.join([str(a) for a in bbox])}\n")


def main(input_dir: str) -> None:
    """Convert dataset main function.

    Iterates through a directory of XML files, converting each to YOLO format
    and saving the result to a specified output directory.

    :param input_dir: Path to the input directory containing input XML files.
    :param output_dir: Path to the output directory.
    """
    class_mapping = {
    "Cng": 0,
    "Rickshaw": 1,
    "Car": 2,
    "Bus": 3,
    "Bike": 4,
    "Mini-Truck": 5,
    "Truck": 6
}


    for folders in Path(input_dir).iterdir():
        if not folders.is_dir():
            continue

        for xml_path in folders.glob("*.xml"):
            txt_path = xml_path.with_suffix(".txt")
            xml_to_txt(str(xml_path), str(txt_path), class_mapping)
            print(f"Converted {xml_path} to {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert XML dataset to YOLO format"
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing input XML files"
    )

    args = parser.parse_args()
    main(args.input_dir)