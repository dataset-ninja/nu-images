***nuImages*** was developed by the authors as an addition to the widely acclaimed [nuScenes](https://www.nuscenes.org/nuscenes) dataset. The strength of nuScenes is in the 1000 carefully curated scenes with 3d annotations, which cover many challenging driving situations. nuImages providing 93,000 2d annotated images from a much larger pool of data. A number of related image datasets for autonomous driving were released in the past. The authors believe that nuImages can complement the existing offerings by virtue of its size and because it is part of the bigger nuScenes ecosystem that features 3d cuboids, lidar segmentation labels, 2d boxes, instance masks and 2d segmentation masks.

## Dataset creation

The researchers conducted driving experiments in Boston (Seaport and South Boston) and Singapore (One North, Holland Village, and Queenstown), selecting these cities for their reputation of dense traffic and complex driving scenarios. Emphasis was placed on capturing the diversity across locations, including variations in vegetation, buildings, vehicles, road markings, and traffic directions (right versus left-hand traffic).

From an extensive pool of training data, the authors manually identified 84 logs containing 15 hours of driving data, covering a distance of 242 kilometers at an average speed of 16 km/h. The driving routes were meticulously chosen to encompass a broad spectrum of locations (urban, residential, nature, and industrial), times of day (day and night), and weather conditions (sun, rain, and clouds).

To avoid redundant annotations in the nuScenes dataset, where annotating 1.4 million images with 2D annotations would be impractical, the authors decided to label a more diverse large-scale image dataset from nearly 500 logs (compared to 83 in nuScenes). The resulting set of 93,000 images was selected using two distinct approaches. Active learning techniques were employed to designate approximately 75% of the images as challenging, focusing on the uncertainty of an image-based object detector, with particular attention to rare classes like bicycles. The remaining 25% of the images were uniformly sampled to ensure a representative dataset and mitigate strong biases.

Through meticulous review, some images were discarded due to camera artifacts, excessive darkness, or the inclusion of pedestrians' faces. This careful curation aimed to achieve a diverse dataset in terms of class distribution, spatiotemporal distribution, and varied weather and lighting conditions. The annotated images encompass scenarios involving rain, snow, and nighttime, crucial for autonomous driving applications. Additionally, the authors included six past and six future camera images at 2 Hz for each annotated image, providing a temporal dynamic aspect to the dataset. Consequently, nuImages comprises 93,000 video clips, each containing 13 frames spaced out at 2 Hz.

The authors labeled a total of 93,000 images with instance masks and 2d boxes for 800k foreground objects and 100k semantic segmentation masks.

<img src="https://github.com/dataset-ninja/nu-images/assets/120389559/11becab1-54c5-4c15-b647-229ffe29b9c5" alt="image" width="1000">

<span style="font-size: smaller; font-style: italic;">Example of labeled image.</span>

The foreground objects additionally have attribute annotations such as whether a motorcycle has a rider, the pose of a pedestrian, the activity of a vehicle, flashing emergency lights and whether an animal is flying.

<img src="https://github.com/dataset-ninja/nu-images/assets/120389559/7cad3010-1122-42bd-b063-4c5870f3e1f6" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Attribute frequencies (excl. test set).</span>

An ***attribute description*** is a property of an instance that can change while the category remains the same. Example: a vehicle being parked/stopped/moving, and whether or not a bicycle has a rider. ***supercategory*** is used for the taxonomy of object and surface categories. Example: vehicle -> car.


