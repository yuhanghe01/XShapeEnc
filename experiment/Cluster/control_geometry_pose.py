import numpy as np
import pickle
import sys
import copy
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
sys.path.append('../../')
sys.path.append("../../utils/")
import ShapeGeometryPoseEncoder
import ShapePoseEncoder
import ShapeGeometryEncoder

np.random.seed(42)

SHAPE_IDS = [4, 15, 39, 49]
AREA_IDS = ['topleft', 'topright', 'bottomleft', 'bottomright']

AREA_COLORS = {
    'bottomright': (227/255, 106/255, 107/255, 1.0),     # red
    'topleft': (107/255, 188/255, 108/255, 1.0),     # blue
    'topright': (99/255, 158/255, 202/255, 1.0),   # green
    'bottomleft': (255/255, 166/255, 98/255, 1.0),  # brown
}

#get geometric shapes
def get_shape_geometries():
    with open(f'data/shape_info.pkl', 'rb') as f:
        shape_info = pickle.load(f)
    return shape_info

def get_shape_pose_vecs(area_id, add_scale, num = 4):
    area_pad = 10
    area_range = 100
    scale_range = [0.8, 1.2]

    topleft_x_range = [0+area_pad, area_range//2 - area_pad]
    topleft_y_range = [0+area_pad, area_range//2 - area_pad]

    topright_x_range = [area_range//2 + area_pad, area_range - area_pad]
    topright_y_range = [0+area_pad, area_range//2 - area_pad]

    bottomleft_x_range = [0+area_pad, area_range//2 - area_pad]
    bottomleft_y_range = [area_range//2 + area_pad, area_range - area_pad]

    bottomright_x_range = [area_range//2 + area_pad, area_range - area_pad]
    bottomright_y_range = [area_range//2 + area_pad, area_range - area_pad]

    if area_id == 'topleft':
        x_coords = np.random.uniform(topleft_x_range[0], topleft_x_range[1], num)
        y_coords = np.random.uniform(topleft_y_range[0], topleft_y_range[1], num)
    elif area_id == 'topright':
        x_coords = np.random.uniform(topright_x_range[0], topright_x_range[1], num)
        y_coords = np.random.uniform(topright_y_range[0], topright_y_range[1], num)
    elif area_id == 'bottomleft':
        x_coords = np.random.uniform(bottomleft_x_range[0], bottomleft_x_range[1], num)
        y_coords = np.random.uniform(bottomleft_y_range[0], bottomleft_y_range[1], num)
    elif area_id == 'bottomright':
        x_coords = np.random.uniform(bottomright_x_range[0], bottomright_x_range[1], num)
        y_coords = np.random.uniform(bottomright_y_range[0], bottomright_y_range[1], num)
    else:
        raise ValueError("area_id must be one of 'topleft', 'topright', 'bottomleft', 'bottomright'")

    scales = np.random.uniform(scale_range[0], scale_range[1], num)

    #final coord: [X, Y, Scale]
    if add_scale:
        coords = np.stack([x_coords/area_range, y_coords/area_range, scales], axis=1).squeeze()
    else:
        coords = np.stack([x_coords/area_range, y_coords/area_range], axis=1).squeeze()

    return coords

def get_spatially_grounded_shape(shape_info, add_scale = True):
    topleft_shape = copy.deepcopy(shape_info)
    topright_shape = copy.deepcopy(shape_info)
    bottomleft_shape = copy.deepcopy(shape_info)
    bottomright_shape = copy.deepcopy(shape_info)

    for shape_id in SHAPE_IDS:
        topleft_shape[shape_id]['pose_vec'] = get_shape_pose_vecs('topleft', add_scale=add_scale, num=1)
        topright_shape[shape_id]['pose_vec'] = get_shape_pose_vecs('topright', add_scale=add_scale, num=1)
        bottomleft_shape[shape_id]['pose_vec'] = get_shape_pose_vecs('bottomleft', add_scale=add_scale, num=1)
        bottomright_shape[shape_id]['pose_vec'] = get_shape_pose_vecs('bottomright', add_scale=add_scale, num=1)

    return topleft_shape, topright_shape, bottomleft_shape, bottomright_shape

def run_tSNE_clustering(feat_list, cart_masks, area_ids, shape_ids, beta, method_name='XShapeEnc'):
    """
    Run t-SNE clustering and visualize with original cartesian shape marks.
    
    Args:
        feat_list: List of encoded features
        cart_masks: List of cartesian masks for each feature
        area_ids: List of area IDs ('topleft', 'topright', etc.) for each feature
        shape_ids: List of shape IDs for each feature
        beta: Beta parameter for filename
        method_name: Name of the method for filename
    """
    feat_array = np.array(feat_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=100)
    feat_2d = tsne.fit_transform(feat_array)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    
    # Fixed zoom factor for consistent shape size across all visualizations
    zoom_factor = 0.15
    
    for i in range(len(feat_list)):
        area_id = area_ids[i]
        shape_id = shape_ids[i]
        cart_mask = cart_masks[i]
        color = AREA_COLORS[area_id]  # Already RGBA tuple with transparency
        mask_rgba = np.zeros((*cart_mask.shape, 4))
        mask_rgba[cart_mask > 0.5] = color
        mask_rgba[cart_mask <= 0.5] = [1, 1, 1, 0]  # Transparent background
        
        # Create an OffsetImage with the shape
        imagebox = OffsetImage(mask_rgba, zoom=zoom_factor)
        imagebox.image.axes = ax
        
        # Place the shape at the t-SNE coordinates
        ab = AnnotationBbox(imagebox, (feat_2d[i, 0], feat_2d[i, 1]),
                           frameon=False, pad=0)
        ax.add_artist(ab)
    
    # Calculate axis limits
    x_range = feat_2d[:, 0].max() - feat_2d[:, 0].min()
    y_range = feat_2d[:, 1].max() - feat_2d[:, 1].min()
    
    # Remove axis labels and ticks but keep grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)  # Remove tick marks
    
    # Remove boundary box (spines) but keep grid
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Adjust axis limits to give some padding
    x_pad = x_range * 0.15
    y_pad = y_range * 0.15
    ax.set_xlim(feat_2d[:, 0].min() - x_pad, feat_2d[:, 0].max() + x_pad)
    ax.set_ylim(feat_2d[:, 1].min() - y_pad, feat_2d[:, 1].max() + y_pad)
    
    plt.tight_layout()
    # Save as PDF with high DPI but optimized file size
    plt.savefig(f'{method_name}_beta_{beta}.pdf', dpi=300, 
                bbox_inches='tight', pad_inches=0.1, 
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved visualization to {method_name}_beta_{beta}.pdf")

def run_addition_cluster(beta=0.5):
    lam = 0.2
    encode_len = 512
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = get_spatially_grounded_shape(shape_info, add_scale=False)

    shape_geo_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(encode_len=encode_len, res=300, lam=lam)
    shape_pose_encoder = ShapePoseEncoder.ShapePoseEncoder(res = 300, 
                                                           K=2,
                                                           L=512,
                                                           m_p=10,
                                                           seed_sigma=0.1)
    
    # Store shapes by area for easy access
    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }

    shape_geometry_masks = list()
    shape_pose_vecs = list()
    shape_cart_masks = list()  # Store cartesian masks for visualization
    shape_area_ids = list()    # Store area IDs for coloring
    shape_id_list = list()     # Store shape IDs for labeling
    
    for shape_id in SHAPE_IDS:
        for area_id in AREA_IDS:
            area_shape = area_shapes[area_id]
            shape_geometry_masks.append(area_shape[shape_id]['polar_mask'])
            shape_pose_vecs.append(area_shape[shape_id]['pose_vec'])
            shape_cart_masks.append(area_shape[shape_id]['cart_mask'])
            shape_area_ids.append(area_id)
            shape_id_list.append(shape_id)
    
    feat_list = list()

    for shape_mask, pose_vec in zip(shape_geometry_masks, shape_pose_vecs):
        geo_encoding = shape_geo_encoder.encode(shape_mask.astype(np.float64))
        pose_encoding = shape_pose_encoder.encode(pose_vec.astype(np.float64), just_return_coeff=True)
        combined_encoding = np.array(geo_encoding, np.float32) + beta * np.array(pose_encoding, np.float32)
        combined_encoding = combined_encoding.tolist()
        feat_list.append(combined_encoding)

    run_tSNE_clustering(feat_list, shape_cart_masks, shape_area_ids, shape_id_list, beta=beta, method_name='Addition')

def run_concate_cluster(beta=0.5):
    lam = 0.2
    encode_len = 256
    pose_vec_len = 2
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = get_spatially_grounded_shape(shape_info, add_scale=False)

    shape_geo_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(encode_len=encode_len, res=300, lam=lam)
    shape_pose_encoder = ShapePoseEncoder.ShapePoseEncoder(res = 300, 
                                                           K=pose_vec_len,
                                                           L=encode_len,
                                                           m_p=10,
                                                           seed_sigma=0.1)
    
    # Store shapes by area for easy access
    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }

    shape_geometry_masks = list()
    shape_pose_vecs = list()
    shape_cart_masks = list()  # Store cartesian masks for visualization
    shape_area_ids = list()    # Store area IDs for coloring
    shape_id_list = list()     # Store shape IDs for labeling
    
    for shape_id in SHAPE_IDS:
        for area_id in AREA_IDS:
            area_shape = area_shapes[area_id]
            shape_geometry_masks.append(area_shape[shape_id]['polar_mask'])
            shape_pose_vecs.append(area_shape[shape_id]['pose_vec'])
            shape_cart_masks.append(area_shape[shape_id]['cart_mask'])
            shape_area_ids.append(area_id)
            shape_id_list.append(shape_id)
    
    feat_list = list()

    for shape_mask, pose_vec in zip(shape_geometry_masks, shape_pose_vecs):
        geo_encoding = shape_geo_encoder.encode(shape_mask.astype(np.float64))
        if len(geo_encoding) > encode_len:
            geo_encoding = geo_encoding[:encode_len]
        pose_encoding = shape_pose_encoder.encode(pose_vec.astype(np.float64), just_return_coeff=True)
        pose_encoding = beta * np.array(pose_encoding, np.float32)
        pose_encoding = pose_encoding.tolist()
        combined_encoding = geo_encoding + pose_encoding
        assert len(combined_encoding) == 2*encode_len, f"Combined encoding length {len(combined_encoding)} does not match expected {2*encode_len}"
        feat_list.append(combined_encoding)

    run_tSNE_clustering(feat_list, shape_cart_masks, shape_area_ids, shape_id_list, beta=beta, method_name='Concate')


def run_XShapeEnc_cluster(beta=0.5):
    lam = 0.2
    encode_len = 512
    pose_vec_len = 2
    shape_info = get_shape_geometries()
    topleft_shape, topright_shape, bottomleft_shape, bottomright_shape = get_spatially_grounded_shape(shape_info, add_scale=False)

    # Store shapes by area for easy access
    area_shapes = {
        'topleft': topleft_shape,
        'topright': topright_shape,
        'bottomleft': bottomleft_shape,
        'bottomright': bottomright_shape,
    }

    shape_geometry_masks = list()
    shape_pose_vecs = list()
    shape_cart_masks = list()  # Store cartesian masks for visualization
    shape_area_ids = list()    # Store area IDs for coloring
    shape_id_list = list()     # Store shape IDs for labeling
    
    for shape_id in SHAPE_IDS:
        for area_id in AREA_IDS:
            area_shape = area_shapes[area_id]
            shape_geometry_masks.append(area_shape[shape_id]['polar_mask'])
            shape_pose_vecs.append(area_shape[shape_id]['pose_vec'])
            shape_cart_masks.append(area_shape[shape_id]['cart_mask'])
            shape_area_ids.append(area_id)
            shape_id_list.append(shape_id)

    geo_pose_encoder = ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder(
        n_max=20,
        res=300,
        lam=lam,
        encode_len=encode_len,
        pose_vec_len=pose_vec_len,
        seed_sigma=0.1,
        beta=beta,
    )

    feat_list = list()
    for shape_mask, pose_vec in zip(shape_geometry_masks, shape_pose_vecs):
        encoding = geo_pose_encoder.encode(shape_mask.astype(np.float64), 
                                           pose_vec.astype(np.float64),
                                           run_freqprop=True, 
                                           mask_in_euclidean=False)
        feat_list.append(encoding)

    run_tSNE_clustering(feat_list, shape_cart_masks, shape_area_ids, shape_id_list, beta=beta, method_name='XShapeEnc')


def main():
    # beta_values = [0.2, 1.0, 1.8]
    # for beta in beta_values:
    #     print(f'Running clustering with beta={beta}...')
    #     run_XShapeEnc_cluster(beta=beta)

    alpha_values = [0.01, 0.04, 0.06]
    for alpha in alpha_values:
        print(f'Running clustering with alpha={alpha}...')
        run_addition_cluster(beta=alpha)
        run_concate_cluster(beta=alpha)

if __name__ == '__main__':
    main()