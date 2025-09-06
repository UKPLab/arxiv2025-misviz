# 📊 Misviz and Misviz-synth datasets 

The Misviz and Misviz-synth datasets are made available under a **CC-BY-SA-4.0** license.

### Misviz-synth

Misviz-synth contains 81,814 synthetic visualizations generated with Matplotlib. It is split into a train, validation, and test set.

- *data/misviz_synth/misviz_synth.json* contains the task labels and metadata
- The visualizations, the underlying data tables, and the axis metadata can be downloaded from [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4782)

Each record contains the following items: 

- `image_path`: local path to the image
- `chart_type`: a list of chart types present in the visualization
- `misleader`: a list of misleaders present in the visualization. If it is empty, the visualization is not misleading.
- `split`: the dataset split (train, train small, val, or test)
- `table_data_path`: local path to the file containing the underlying data table
- `axis_data_path`: local path to the file containing the axis metadata
- `plotting_mechanism`: the template used to generate the visualization
- `chart_title`: title of the chart
- `data_origin`: the source of the underlying data table
- `original_table_title`: the name of the original underlying data table
- `column_types`: a dictionary with table column names as keys and lists containing the types suitable for that column as values

### Misviz 

Misviz contains 2,604 real-world visualizations collected from various websites. It is split into a dev, validation, and test set.

- *data/misviz/misviz.json* contains the task labels and metadata
- The visualizations can be downloaded from the web using the following script

```python
$ python data/download_misviz_images.py --use_wayback 0
```

```use_wayback``` is a paramater to decide whether the image is scraped from the original URL (0) or from an archived version of the URL on the Wayback Machine (1).
The archived URL serves as a backup.

Please contact  `jonathan.tonglet@tu-darmstadt.de` if you face issues downloading the images of Misviz.

Each record contains the following items: 

- `image_path`: local path to the image
- `image_url`: URL of the image
- `chart_type`: a list of chart types present in the visualization
- `misleader`: a list of misleaders present in the visualization. If it is empty, the visualization is not misleading.
- `wayback_image_url`: URL of the archived image on the Wayback Machine
- `split`: the dataset split (train, train small, val, or test)


