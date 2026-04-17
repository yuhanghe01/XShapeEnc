import ShapeCorpusGen
import yaml
import os

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    save_dir = config['COMPLEX_SHAPE_GEN']['shape_save_dir']
    os.makedirs(save_dir, exist_ok=True)
    shape_corpus_gen = ShapeCorpusGen.XShapeGen(config)
    for depth in [1,2,3,4,5,6,7,8,9,10]:
        print(f"Generating shapes with depth {depth}...")
        sub_save_dir = os.path.join(save_dir, f'depth_{depth}')
        os.makedirs(sub_save_dir, exist_ok=True)
        shape_dict = shape_corpus_gen.create_shape_one_depth(depth=depth, gen_num=config['COMPLEX_SHAPE_GEN']['shape_num'])