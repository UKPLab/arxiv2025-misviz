## Datasets

### Misviz-synth

- *data/misviz_synth/misviz_synth.json* contains the task labels and metadata
- The visualizations, the underlying data tables, and the axis metadata can be downloaded from [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4782)

### Misviz 

- *data/misviz/misviz.json* contains the task labels and metadata
- The visualizations can be downloaded from the web using the following script

```python
$ python data/download_misviz_images.py --use_wayback 0
```

```use_wayback``` is a paramater to decide whether the image is scraped from the original URL (0) or from an archived version of the URL on the Wayback Machine (1).
The archived URL serves as a backup.
