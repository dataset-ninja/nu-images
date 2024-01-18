**nuImages: A Multimodal Dataset for Autonomous Driving** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the automotive industry. 

The dataset consists of 93476 images with 2122939 labeled objects belonging to 25 different classes including *driveable surface*, *car*, *adult*, and other: *truck*, *trafficcone*, *barrier*, *ego*, *motorcycle*, *bicycle*, *rigid*, *construction worker*, *construction*, *bicycle rack*, *pushable pullable*, *trailer*, *debris*, *child*, *personal mobility*, *police officer*, *stroller*, *animal*, *bendy*, *police*, *ambulance*, and *wheelchair*.

Images in the nuImages dataset have pixel-level instance segmentation and bounding box annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation task (only one mask for every class). There are 11858 (13% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *train* (67279 images), *val* (16445 images), and *test* (9752 images). Additionally, labels have ***category*** tag, ***attribute desc***, ***attribute***, ***category desc***. Also every image contains information about its ***sensor***. The dataset was released in 2020 by the NuTonomy.

Here are the visualized examples for the classes:

[Dataset classes](https://github.com/dataset-ninja/nu-images/raw/main/visualizations/classes_preview.webm)
