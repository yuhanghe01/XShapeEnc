## `XShapeCorpus` benchmark generation

1. step 1: edit `config.yaml` to specify the shape generation configuration. Typical configuration include:

    ```yaml
    # config.yaml
    pose_gen_translate_range: [-100, 100]
    pose_scale_range: [1.5, 100.0]
    canvas_size: 256
    shape_mask_resolution: 300
    shape_save_dir: 'shape_corpus'
    op_num: 15
    shape_num: 100
    ```
2. step 2: run `main.py` to generate the corpus.

   ```python
   python main.py
   ```