from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "nuImages"
PROJECT_NAME_FULL: str = "nuImages: A Multimodal Dataset for Autonomous Driving"
HIDE_DATASET = True  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.CC_BY_NC_SA_4_0(source_url="https://www.nuscenes.org/terms-of-use")
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Industry.Automotive()]
CATEGORY: Category = Category.SelfDriving()

CV_TASKS: List[CVTask] = [
    CVTask.InstanceSegmentation(),
    CVTask.SemanticSegmentation(),
    CVTask.ObjectDetection(),
]
ANNOTATION_TYPES: List[AnnotationType] = [
    AnnotationType.InstanceSegmentation(),
    AnnotationType.ObjectDetection(),
]

RELEASE_DATE: Optional[str] = "2020-04-30"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://www.nuscenes.org/nuimages"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 12637639
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/nu-images"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = "https://www.nuscenes.org/nuimages#download"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = {
    "adult": [230, 25, 75],
    "ambulance": [60, 180, 75],
    "animal": [255, 225, 25],
    "barrier": [0, 130, 200],
    "bendy": [245, 130, 48],
    "bicycle": [145, 30, 180],
    "bicycle rack": [70, 240, 240],
    "car": [240, 50, 230],
    "child": [210, 245, 60],
    "construction": [250, 190, 212],
    "construction worker": [0, 128, 128],
    "debris": [220, 190, 255],
    "driveable surface": [170, 110, 40],
    "motorcycle": [255, 250, 200],
    "personal mobility": [128, 0, 0],
    "police": [170, 255, 195],
    "police officer": [128, 128, 0],
    "pushable pullable": [255, 215, 180],
    "rigid": [0, 0, 128],
    "stroller": [0, 255, 0],
    "trafficcone": [160, 82, 45],
    "trailer": [144, 238, 144],
    "truck": [255, 192, 203],
    "wheelchair": [244, 164, 96],
    "ego": [30, 30, 30],
}
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = "https://arxiv.org/pdf/1903.11027.pdf"
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = {
    "GitHub": "https://github.com/nutonomy/nuscenes-devkit"
}

CITATION_URL: Optional[str] = None
AUTHORS: Optional[List[str]] = [
    "Holger Caesar",
    "Varun Bankiti",
    "Alex H. Lang",
    "Sourabh Vora",
    "Venice Erin Liong",
    "Qiang Xu",
    "Anush Krishnan",
    "Yu Pan",
    "Giancarlo Baldan",
    "Oscar Beijbom",
]
AUTHORS_CONTACTS: Optional[List[str]] = ["nuscenes@nutonomy.com"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "NuTonomy"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://motional.com/"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Additionally, labels have ***supercategory*** tag, ***attribute description***, ***attribute name***, ***description***. Also every image contains information about its ***camera***. Explore it in supervisely labeling tool"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
