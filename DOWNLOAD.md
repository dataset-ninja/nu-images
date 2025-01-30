Dataset **nuImages** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzMzNDhfbnVJbWFnZXMvbnVpbWFnZXMtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiRE1JU3Fld0tsOStIeXQvVmo0dkFRQzhTMFpyMGdvejlYaVg1ZW8wdUVrND0ifQ==)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='nuImages', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://www.nuscenes.org/nuimages#download).