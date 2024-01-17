import base64
import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from pycocotools import mask as cocomask
from supervisely.io.fs import file_exists, get_file_name, get_file_name_with_ext
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    images_path = "/home/alex/DATASETS/TODO/nuImages/nuimages-v1.0-all-samples"
    train_anns_path = "/home/alex/DATASETS/TODO/nuImages/nuimages-v1.0-all-metadata/v1.0-train"
    val_anns_path = "/home/alex/DATASETS/TODO/nuImages/nuimages-v1.0-all-metadata/v1.0-val"
    test_anns_path = "/home/alex/DATASETS/TODO/nuImages/nuimages-v1.0-all-metadata/v1.0-test"
    batch_size = 30

    ds_name_to_data = {"train": train_anns_path, "val": val_anns_path, "test": test_anns_path}

    def mask_decode(mask: dict) -> np.ndarray:
        new_mask = mask.copy()
        new_mask["counts"] = base64.b64decode(mask["counts"])
        return cocomask.decode(new_mask)

    def create_ann(image_subpath):
        labels = []

        subfolder_value = image_subpath.split("/")[-2]
        subfolder = sly.Tag(subfolder_meta, value=subfolder_value)

        image_name = get_file_name_with_ext(image_subpath)
        img_height = int(name_to_shape[image_name][0])
        img_wight = int(name_to_shape[image_name][1])

        ann_data = im_path_to_data[image_subpath]
        for curr_ann_data in ann_data:
            tags = []
            obj_class_data = token_to_category_data[curr_ann_data[2]]
            obj_class = name_to_class[obj_class_data[0]]
            description_value = obj_class_data[1][:254]
            class_description = sly.Tag(class_descr_meta, value=description_value)
            tags.append(class_description)

            supercategory_value = class_to_supercategory.get(obj_class_data[0])
            if supercategory_value is not None:
                supercategory = sly.Tag(supercategory_meta, value=supercategory_value)
                tags.append(supercategory)

            for attribute_data in curr_ann_data[3]:
                curr_attribute_data = token_to_attribute_data[attribute_data]
                description_value = curr_attribute_data[1][:254]
                attribute_descr = sly.Tag(attribute_descr_meta, value=description_value)
                tags.append(attribute_descr)

                name_value = curr_attribute_data[0]
                attribute_name = sly.Tag(attribute_name_meta, value=name_value)
                tags.append(attribute_name)

            mask_coco = curr_ann_data[1]
            if mask_coco is not None:
                mask = mask_decode(mask_coco)
                if len(np.unique(mask)) > 1:
                    polygons = sly.Bitmap(mask).to_contours()
                    if polygons is not None:
                        for polygon in polygons:
                            label = sly.Label(polygon, obj_class, tags=tags)
                            labels.append(label)

            left = curr_ann_data[0][0]
            right = curr_ann_data[0][2]
            top = curr_ann_data[0][1]
            bottom = curr_ann_data[0][3]
            rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
            label = sly.Label(rectangle, obj_class, tags=tags)
            labels.append(label)

        surface_data = im_path_to_surface[image_subpath]
        for curr_surface_data in surface_data:
            tags = []
            obj_class_data = token_to_category_data[curr_surface_data[1]]
            obj_class = name_to_class[obj_class_data[0]]
            description_value = obj_class_data[1][:254]
            class_description = sly.Tag(class_descr_meta, value=description_value)
            tags.append(class_description)

            supercategory_value = class_to_supercategory[obj_class_data[0]]
            supercategory = sly.Tag(supercategory_meta, value=supercategory_value)
            tags.append(supercategory)

            mask_coco = curr_surface_data[0]
            if mask_coco is not None:
                mask = mask_decode(mask_coco)
                if len(np.unique(mask)) > 1:
                    polygons = sly.Bitmap(mask).to_contours()
                    if polygons is not None:
                        for polygon in polygons:
                            label = sly.Label(polygon, obj_class, tags=tags)
                            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[subfolder])

    animal = sly.ObjClass("animal", sly.AnyGeometry)
    driveable_surface = sly.ObjClass("driveable surface", sly.AnyGeometry)
    adult = sly.ObjClass("adult", sly.AnyGeometry)
    child = sly.ObjClass("child", sly.AnyGeometry)
    construction_worker = sly.ObjClass("construction worker", sly.AnyGeometry)
    personal_mobility = sly.ObjClass("personal mobility", sly.AnyGeometry)
    police_officer = sly.ObjClass("police officer", sly.AnyGeometry)
    stroller = sly.ObjClass("stroller", sly.AnyGeometry)
    wheelchair = sly.ObjClass("wheelchair", sly.AnyGeometry)
    barrier = sly.ObjClass("barrier", sly.AnyGeometry)
    debris = sly.ObjClass("debris", sly.AnyGeometry)
    pushable_pullable = sly.ObjClass("pushable pullable", sly.AnyGeometry)
    trafficcone = sly.ObjClass("trafficcone", sly.AnyGeometry)
    bicycle_rack = sly.ObjClass("bicycle rack", sly.AnyGeometry)
    bicycle = sly.ObjClass("bicycle", sly.AnyGeometry)
    bendy = sly.ObjClass("bendy", sly.AnyGeometry)
    rigid = sly.ObjClass("rigid", sly.AnyGeometry)
    car = sly.ObjClass("car", sly.AnyGeometry)
    construction = sly.ObjClass("construction", sly.AnyGeometry)
    ego = sly.ObjClass("ego", sly.AnyGeometry)
    ambulance = sly.ObjClass("ambulance", sly.AnyGeometry)
    police = sly.ObjClass("police", sly.AnyGeometry)
    motorcycle = sly.ObjClass("motorcycle", sly.AnyGeometry)
    trailer = sly.ObjClass("trailer", sly.AnyGeometry)
    truck = sly.ObjClass("truck", sly.AnyGeometry)

    name_to_class = {
        "animal": animal,
        "flat.driveable_surface": driveable_surface,
        "human.pedestrian.adult": adult,
        "human.pedestrian.child": child,
        "human.pedestrian.construction_worker": construction_worker,
        "human.pedestrian.personal_mobility": personal_mobility,
        "human.pedestrian.police_officer": police_officer,
        "human.pedestrian.stroller": stroller,
        "human.pedestrian.wheelchair": wheelchair,
        "movable_object.barrier": barrier,
        "movable_object.debris": debris,
        "movable_object.pushable_pullable": pushable_pullable,
        "movable_object.trafficcone": trafficcone,
        "static_object.bicycle_rack": bicycle_rack,
        "vehicle.bicycle": bicycle,
        "vehicle.bus.bendy": bendy,
        "vehicle.bus.rigid": rigid,
        "vehicle.car": car,
        "vehicle.construction": construction,
        "vehicle.ego": ego,
        "vehicle.emergency.ambulance": ambulance,
        "vehicle.emergency.police": police,
        "vehicle.motorcycle": motorcycle,
        "vehicle.trailer": trailer,
        "vehicle.truck": truck,
    }

    class_to_supercategory = {
        "flat.driveable_surface": "flat",
        "human.pedestrian.adult": "human, pedestrian",
        "human.pedestrian.child": "human, pedestrian",
        "human.pedestrian.construction_worker": "human, pedestrian",
        "human.pedestrian.personal_mobility": "human, pedestrian",
        "human.pedestrian.police_officer": "human, pedestrian",
        "human.pedestrian.stroller": "human, pedestrian",
        "human.pedestrian.wheelchair": "human, pedestrian",
        "movable_object.barrier": "movable object",
        "movable_object.debris": "movable object",
        "movable_object.pushable_pullable": "movable object",
        "movable_object.trafficcone": "movable object",
        "static_object.bicycle_rack": "static object",
        "vehicle.bicycle": "vehicle",
        "vehicle.bus.bendy": "vehicle bus",
        "vehicle.bus.rigid": "vehicle bus",
        "vehicle.car": "vehicle",
        "vehicle.construction": "vehicle",
        "vehicle.ego": "vehicle",
        "vehicle.emergency.ambulance": "vehicle emergency",
        "vehicle.emergency.police": "vehicle emergency",
        "vehicle.motorcycle": "vehicle",
        "vehicle.trailer": "vehicle",
        "vehicle.truck": "vehicle",
    }

    subfolder_meta = sly.TagMeta("camera", sly.TagValueType.ANY_STRING)
    class_descr_meta = sly.TagMeta("description", sly.TagValueType.ANY_STRING)
    attribute_name_meta = sly.TagMeta("attribute name", sly.TagValueType.ANY_STRING)
    attribute_descr_meta = sly.TagMeta("attribute description", sly.TagValueType.ANY_STRING)
    supercategory_meta = sly.TagMeta("supercategory", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(name_to_class.values()),
        tag_metas=[
            subfolder_meta,
            class_descr_meta,
            attribute_name_meta,
            attribute_descr_meta,
            supercategory_meta,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, anns_path in ds_name_to_data.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        name_to_shape = {}
        sample_token_to_im_path = {}
        im_path_to_data = defaultdict(list)
        images_subpathes = []

        sample_data = load_json_file(os.path.join(anns_path, "sample_data.json"))
        for curr_sample_data in sample_data:
            im_name = get_file_name_with_ext(curr_sample_data["filename"])

            images_subpathes.append(curr_sample_data["filename"])

            name_to_shape[im_name] = (curr_sample_data["height"], curr_sample_data["width"])
            sample_token_to_im_path[curr_sample_data["token"]] = curr_sample_data["filename"]

        object_ann = load_json_file(os.path.join(anns_path, "object_ann.json"))
        for curr_object_ann in object_ann:
            curr_sample_token = curr_object_ann["sample_data_token"]
            im_path = sample_token_to_im_path.get(curr_sample_token)
            if im_path is not None:
                im_path_to_data[im_path].append(
                    [
                        curr_object_ann["bbox"],
                        curr_object_ann["mask"],
                        curr_object_ann["category_token"],
                        curr_object_ann["attribute_tokens"],
                    ]
                )

        token_to_attribute_data = {}
        attribute = load_json_file(os.path.join(anns_path, "attribute.json"))
        for curr_attribute in attribute:
            token_to_attribute_data[curr_attribute["token"]] = (
                curr_attribute["name"],
                curr_attribute["description"],
            )

        token_to_category_data = {}
        category = load_json_file(os.path.join(anns_path, "category.json"))
        for curr_category in category:
            token_to_category_data[curr_category["token"]] = (
                curr_category["name"],
                curr_category["description"],
            )

        im_path_to_surface = defaultdict(list)
        surface = load_json_file(os.path.join(anns_path, "surface_ann.json"))
        for curr_surface in surface:
            curr_sample_token = curr_surface["sample_data_token"]
            im_path = sample_token_to_im_path.get(curr_sample_token)
            if im_path is not None:
                im_path_to_surface[im_path].append(
                    [
                        curr_surface["mask"],
                        curr_surface["category_token"],
                    ]
                )

        temp = []  # del swaps folder data
        for subpath in images_subpathes:
            if subpath[:7] == "samples":
                temp.append(subpath)
        images_subpathes = temp

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_subpathes))

        for images_subpathes_batch in sly.batched(images_subpathes, batch_size=batch_size):
            img_pathes_batch = []
            images_names_batch = []
            exist_images_subpathes_batch = []
            for image_subpath in images_subpathes_batch:
                curr_im_path = os.path.join(images_path, image_subpath)
                if file_exists(curr_im_path):
                    img_pathes_batch.append(curr_im_path)
                    images_names_batch.append(get_file_name_with_ext(image_subpath))
                    exist_images_subpathes_batch.append(image_subpath)

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_subpath) for image_subpath in exist_images_subpathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
