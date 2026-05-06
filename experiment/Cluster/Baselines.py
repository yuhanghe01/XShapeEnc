import numpy as np
import cv2
from pyefd import elliptic_fourier_descriptors, normalize_efd
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt


class Baseline:

    _resnet18_model = None
    _resnet18_device = None
    _vit_model = None
    _vit_device = None
    _clip_model = None
    _clip_preprocess = None

    @staticmethod
    def angular_sweep_encoding(mask: np.ndarray, encode_len: int) -> list:
        """
        Baseline angular encoding for an arbitrary shape mask in the unit disk.
        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary (or 0/1-valued) shape mask. The shape is assumed to lie within
            the unit disk, and the origin (0,0) is assumed to be at the center of
            the array.
        encode_len : int
            Desired length of the final encoding vector.
        Returns
        -------
        feat : list, shape (encode_len,)
            Real-valued encoding feature vector built by sweeping rays from the
            origin, ordered angle-wise (0 rad to 2π, anticlockwise), and for each
            angle, near-to-far boundary intersections.
            If the number of collected distances >= encode_len, it's truncated;
            otherwise, the remainder is padded with zeros.
        """
        mask = (mask > 0).astype(np.uint8)
        H, W = mask.shape

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        n_angles = max(encode_len, 1)

        n_r = max(H, W) * 4
        r_samples = np.linspace(0.0, 1.0, n_r, endpoint=True)

        all_distances = []

        thetas = np.linspace(0.0, 2.0 * np.pi, num=n_angles, endpoint=False)

        for theta in thetas:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            xs = r_samples * cos_t
            ys = r_samples * sin_t

            j_float = (xs + 1.0) * 0.5 * (W - 1)
            i_float = (1.0 - (ys + 1.0) * 0.5) * (H - 1)

            j_idx = np.clip(np.round(j_float).astype(int), 0, W - 1)
            i_idx = np.clip(np.round(i_float).astype(int), 0, H - 1)

            vals = mask[i_idx, j_idx].astype(np.uint8)
            inside = vals
            changes = np.where(inside[:-1] != inside[1:])[0]

            if len(changes) == 0:
                continue

            for idx in changes:
                r_int = 0.5 * (r_samples[idx] + r_samples[idx + 1])
                all_distances.append(r_int)

        all_distances = np.array(all_distances, dtype=np.float32)

        if all_distances.size >= encode_len:
            feat = all_distances[:encode_len]
        else:
            feat = np.zeros(encode_len, dtype=np.float32)
            feat[: all_distances.size] = all_distances

        return feat.tolist()

    @staticmethod
    def sinusoidal_encoding_1d(values, encode_len: int) -> list:
        """
        Apply sinusoidal positional encoding to 1D scalar values and aggregate via mean pooling.

        Parameters
        ----------
        values : np.ndarray, shape (N,)
            Array of N scalar values.
        encode_len : int
            Desired length of the output encoding vector. Must be even.

        Returns
        -------
        encoding : list, shape (encode_len,)
            Mean-pooled sinusoidal encoding across all N values.
        """
        values = np.asarray(values, dtype=np.float64)
        assert values.ndim == 1, "values must be 1D array (N,)"
        assert encode_len % 2 == 0, "encode_len must be even"

        N = values.shape[0]

        if N == 0:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        num_freqs = encode_len // 2

        i = np.arange(num_freqs, dtype=np.float64)
        freq_scale = 1.0 / (10000.0 ** (2 * i / encode_len))

        values_expanded = values[:, np.newaxis]
        scaled_values = values_expanded * freq_scale[np.newaxis, :]

        encodings = np.zeros((N, encode_len), dtype=np.float64)
        encodings[:, 0::2] = np.sin(scaled_values)
        encodings[:, 1::2] = np.cos(scaled_values)

        encoding = encodings.mean(axis=0)

        return encoding.tolist()

    @staticmethod
    def sinusoidal_encoding(points: np.ndarray, encode_len: int) -> list:
        """
        Apply sinusoidal positional encoding to a set of 2D points and aggregate.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
            Input 2D coordinates to encode.
        encode_len : int
            Desired length of the final encoding vector.

        Returns
        -------
        enc : list, shape (encode_len,)
            Aggregated sinusoidal encoding of the point set.
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        N, D = points.shape

        num_freqs = encode_len // (2 * D)

        if num_freqs == 0:
            return np.zeros(encode_len, dtype=np.float32).tolist()

        i_vals = np.arange(num_freqs, dtype=np.float32)
        freqs = 1.0 / (10000.0 ** (2.0 * i_vals / encode_len))

        encodings = []
        for d in range(D):
            coord = points[:, d:d+1]
            angles = coord * freqs[np.newaxis, :]
            encodings.append(np.sin(angles))
            encodings.append(np.cos(angles))

        enc = np.concatenate(encodings, axis=-1)
        enc_aggregated = np.mean(enc, axis=0)

        if len(enc_aggregated) > encode_len:
            enc_aggregated = enc_aggregated[:encode_len]
        elif len(enc_aggregated) < encode_len:
            pad_width = encode_len - len(enc_aggregated)
            enc_aggregated = np.pad(enc_aggregated, (0, pad_width), mode='constant', constant_values=0.0)

        return enc_aggregated.tolist()

    @staticmethod
    def distance_transform_encoding(mask: np.ndarray, encode_len: int) -> list:
        """
        Distance transform encoding for a binary shape mask.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary (or 0/1-valued) shape mask.
        encode_len : int
            Number of distance samples to return.

        Returns
        -------
        feat : list, shape (encode_len,)
            Distance transform encoding feature vector.
        """
        mask = (mask > 0).astype(np.uint8)

        # Interior DT: each foreground pixel gets its distance to the nearest boundary.
        # Using (1 - mask) gives 0 for foreground pixels, causing an all-zero sorted prefix.
        dist_transform = distance_transform_edt(mask)

        # Extract only foreground pixel distances (background pixels are 0 and uninformative).
        fg_dists = dist_transform[mask > 0].astype(np.float32)

        if fg_dists.size == 0:
            return [0.] * encode_len

        max_dist = fg_dists.max()
        if max_dist > 0:
            fg_dists = fg_dists / max_dist

        fg_dists = np.sort(fg_dists)

        # Uniformly subsample to exactly encode_len values.
        if fg_dists.size >= encode_len:
            indices = np.linspace(0, fg_dists.size - 1, encode_len, dtype=int)
            feat = fg_dists[indices]
        else:
            feat = np.zeros(encode_len, dtype=np.float32)
            feat[: fg_dists.size] = fg_dists

        return feat.tolist()

    @staticmethod
    def space2vec_shape_encoding(
        mask,
        encode_len=None,
        coord_sample_num=1024,
        frequency_num=16,
        min_radius=0.01,
        max_radius=1.0,
        freq_init="geometric",
        random_state=None,
    ) -> list:
        """
        Training-free shape encoding using Space2Vec-style point encoder Enc(x).

        Args:
            mask : 2D numpy array (H, W), binary (0/1 or False/True).
            encode_len : int or None.
                - If None: return natural dim = 4 * frequency_num.
                - If int: crop or zero-pad the natural encoding to this length.
            coord_sample_num : int, number of foreground pixels to sample.
            frequency_num : int, number of spatial frequencies (scales).
            min_radius, max_radius : float, min/max spatial scale.
            freq_init : "geometric" or "nerf".
            random_state : int or None, seed for reproducibility.

        Returns:
            encoding : list of shape (encode_len or 4*frequency_num,)
        """
        mask = np.asarray(mask)
        assert mask.ndim == 2, "mask must be 2D"

        rng = np.random.RandomState(random_state)

        ys, xs = np.where(mask > 0)
        num_fg = len(xs)

        if num_fg == 0:
            base_dim = 4 * frequency_num
            if encode_len is None:
                encode_len = base_dim
            return np.zeros(encode_len, dtype=np.float64).tolist()

        if num_fg > coord_sample_num:
            idx = rng.choice(num_fg, size=coord_sample_num, replace=False)
            xs = xs[idx]
            ys = ys[idx]
            num_fg = coord_sample_num

        H, W = mask.shape

        xs_norm = (xs + 0.5) / W * 2.0 - 1.0
        ys_norm = (ys + 0.5) / H * 2.0 - 1.0

        coords = np.stack([xs_norm, ys_norm], axis=-1)

        if freq_init == "geometric":
            radii = np.logspace(
                np.log10(min_radius),
                np.log10(max_radius),
                num=frequency_num,
                dtype=np.float64,
            )
            freq_list = 1.0 / radii
        elif freq_init == "nerf":
            freq_list = 2.0 ** np.arange(frequency_num, dtype=np.float64)
        else:
            raise ValueError(f"Unknown freq_init: {freq_init}")

        coords_exp = coords[:, np.newaxis, :]
        freq = freq_list[np.newaxis, :, np.newaxis]
        angle = coords_exp * freq

        sin_part = np.sin(angle)
        cos_part = np.cos(angle)

        feat = np.concatenate([sin_part, cos_part], axis=-1)
        feat_flat = feat.reshape(num_fg, -1)

        encoding = feat_flat.mean(axis=0)

        base_dim = encoding.shape[0]
        if encode_len is None:
            encode_len = base_dim

        if encode_len < base_dim:
            encoding = encoding[:encode_len]
        elif encode_len > base_dim:
            pad_width = encode_len - base_dim
            encoding = np.pad(encoding, (0, pad_width), mode="constant", constant_values=0.0)

        return encoding.tolist()

    @staticmethod
    def shape_distribution_D2_boundary_deterministic(mask, encode_len: int) -> list:
        """
        Deterministic D2 shape distribution baseline.

        Args:
            mask : 2D binary numpy array (H, W)
            encode_len : number of bins in final descriptor

        Returns:
            hist : list of shape (encode_len,), L1-normalized
        """
        mask = (mask > 0).astype(np.uint8)
        H, W = mask.shape

        pad = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=0)

        c     = pad[1:-1, 1:-1]
        up    = pad[:-2, 1:-1]
        down  = pad[2:, 1:-1]
        left  = pad[1:-1, :-2]
        right = pad[1:-1, 2:]

        boundary = (c == 1) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))

        ys, xs = np.where(boundary)
        num_boundary = len(xs)

        if num_boundary == 0:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        max_boundary_samples = 1000
        if num_boundary > max_boundary_samples:
            step = num_boundary // max_boundary_samples
            idx = np.arange(0, num_boundary, step)
            xs = xs[idx[:max_boundary_samples]]
            ys = ys[idx[:max_boundary_samples]]
            num_boundary = len(xs)

        xs_norm = (xs + 0.5) / W
        ys_norm = (ys + 0.5) / H
        coords = np.stack([xs_norm, ys_norm], axis=-1)

        max_points_for_pairs = 200
        if num_boundary > max_points_for_pairs:
            coords = coords[:max_points_for_pairs]
            num_boundary = max_points_for_pairs

        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

        tri_idx = np.triu_indices(num_boundary, k=1)
        dists = dist_matrix[tri_idx]

        if len(dists) == 0:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        max_dist = dists.max()
        if max_dist > 0:
            dists = dists / max_dist
        else:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        hist, _ = np.histogram(
            dists,
            bins=encode_len,
            range=(0.0, 1.0),
            density=False,
        )

        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total

        return hist.tolist()

    @staticmethod
    def eft_encode_shape(binary_mask: np.ndarray, encode_len: int, to_normalise: bool = False) -> list:
        """Encodes a 2D geometric shape from a binary mask using the Elliptic Fourier Transform (EFT)."""
        if binary_mask.dtype != np.uint8:
            binary_mask = (binary_mask * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return [0.] * encode_len

        # Use only the largest contour to keep the output length deterministic.
        main_contour = max(contours, key=cv2.contourArea)

        # order = encode_len // 4 so that flattened EFD (order × 4) == encode_len exactly.
        order = encode_len // 4
        if order == 0 or len(main_contour) <= order:
            return [0.] * encode_len

        efds = elliptic_fourier_descriptors(main_contour.reshape(-1, 2), order=order, normalize=False)
        if to_normalise:
            efds = normalize_efd(efds)

        final_feature_vector = efds.flatten()

        # Safety: truncate or zero-pad to exactly encode_len.
        if len(final_feature_vector) > encode_len:
            final_feature_vector = final_feature_vector[:encode_len]
        elif len(final_feature_vector) < encode_len:
            final_feature_vector = np.pad(final_feature_vector, (0, encode_len - len(final_feature_vector)))

        return final_feature_vector.tolist()

    @staticmethod
    def regular_grid_point_encoding(mask, encode_len: int) -> list:
        """
        Sample points regularly on the shape mask and encode their coordinates.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary (or 0/1-valued) shape mask.
        encode_len : int
            Desired length of the encoding vector. Must be even.
            Will sample encode_len/2 points (each contributes 2 values: x, y).

        Returns
        -------
        encoding : list, shape (encode_len,)
            Concatenated (x, y) coordinates of sampled points in scanline order,
            normalized to [-1, 1].
        """
        mask = (mask > 0).astype(np.uint8)
        H, W = mask.shape

        num_points = encode_len // 2

        if num_points == 0:
            return np.zeros(encode_len, dtype=np.float32).tolist()

        ys, xs = np.where(mask > 0)
        num_fg = len(xs)

        if num_fg == 0:
            return np.zeros(encode_len, dtype=np.float32).tolist()

        if num_fg <= num_points:
            sampled_indices = np.arange(num_fg)
        else:
            sampled_indices = np.linspace(0, num_fg - 1, num_points, dtype=int)

        coords = np.column_stack([ys, xs])
        sort_order = np.lexsort((xs, ys))
        coords_sorted = coords[sort_order]
        sampled_coords = coords_sorted[sampled_indices]

        sampled_xs = (sampled_coords[:, 1] + 0.5) / W * 2.0 - 1.0
        sampled_ys = 1.0 - (sampled_coords[:, 0] + 0.5) / H * 2.0

        encoding = np.zeros(encode_len, dtype=np.float32)
        for i in range(len(sampled_xs)):
            encoding[2*i] = sampled_xs[i]
            encoding[2*i + 1] = sampled_ys[i]

        return encoding.tolist()

    @staticmethod
    def regular_grid_point_encoding_coord(mask, encode_len: int, flatten: bool = True) -> list:
        """
        Sample points regularly on the shape mask and return their coordinates.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary (or 0/1-valued) shape mask.
        encode_len : int
            Desired length of the encoding vector. Must be even.
            Will sample encode_len/2 points (each contributes 2 values: x, y).

        Returns
        -------
        encoding : list, shape (encode_len,)
            Flattened (x, y) coordinates of sampled points, normalized to [-1, 1].
        """
        mask = (mask > 0).astype(np.uint8)
        H, W = mask.shape

        num_points = encode_len // 2

        if num_points == 0:
            return np.zeros(encode_len, dtype=np.float32).tolist()

        ys, xs = np.where(mask > 0)
        num_fg = len(xs)

        if num_fg == 0:
            return np.zeros(encode_len, dtype=np.float32).tolist()

        if num_fg <= num_points:
            sampled_indices = np.arange(num_fg)
        else:
            sampled_indices = np.linspace(0, num_fg - 1, num_points, dtype=int)

        coord_arr = np.column_stack([ys, xs])
        sort_order = np.lexsort((xs, ys))
        coords_sorted = coord_arr[sort_order]
        sampled_coords = coords_sorted[sampled_indices]

        sampled_xs = (sampled_coords[:, 1] + 0.5) / W * 2.0 - 1.0
        sampled_ys = 1.0 - (sampled_coords[:, 0] + 0.5) / H * 2.0

        out = np.zeros((len(sampled_xs), 2), dtype=np.float32)
        for i in range(len(sampled_xs)):
            out[i, 0] = sampled_xs[i]
            out[i, 1] = sampled_ys[i]

        if flatten:
            return out.flatten().tolist()
        else:
            return out.tolist()

    @staticmethod
    def reconstruct_shape_from_efd_v2(encoding_data: dict, num_points: int) -> np.ndarray:
        """
        Reconstructs the original shape mask from a dictionary containing raw EFD data.
        """
        efd_vector = encoding_data['coefficients']
        num_harmonics = len(efd_vector) // 4
        efds = efd_vector.reshape(num_harmonics, 4)

        A0, C0 = encoding_data['A0'], encoding_data['C0']
        T = encoding_data['T']
        image_size = encoding_data['image_size']

        t = np.linspace(0, T, num_points, endpoint=False, dtype=np.float64)
        omega = 2 * np.pi / T

        x_coords = np.full(num_points, A0, dtype=np.float64)
        y_coords = np.full(num_points, C0, dtype=np.float64)

        for n in range(1, num_harmonics + 1):
            a_n, b_n, c_n, d_n = efds[n-1, :]
            x_coords += (a_n * np.cos(n * omega * t) + b_n * np.sin(n * omega * t))
            y_coords += (c_n * np.cos(n * omega * t) + d_n * np.sin(n * omega * t))

        reconstructed_contour = np.vstack((x_coords, y_coords)).T.astype(np.int32)

        reconstructed_mask = np.zeros(image_size, dtype=np.uint8)
        cv2.fillPoly(reconstructed_mask, [reconstructed_contour], 255)

        return reconstructed_mask

    @staticmethod
    def resnet18_shape_encoding(binary_mask, encode_len: int = 512, layer: str = 'avgpool') -> list:
        """
        Extract shape embedding from a binary mask using pre-trained ResNet18.

        Parameters
        ----------
        binary_mask : np.ndarray, shape (H, W)
            Binary shape mask (0s and 1s, or boolean).
        encode_len : int, optional
            Desired length of the output encoding vector. Default is 512.
        layer : str, optional
            Which layer to extract features from: 'avgpool' (512-dim) or 'fc' (1000-dim).

        Returns
        -------
        encoding : list, shape (encode_len,)
            Feature embedding extracted from ResNet18, resized to encode_len.
        """
        if Baseline._resnet18_model is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Baseline._resnet18_device = device
            model = models.resnet18(pretrained=True)
            model = model.to(device)
            model.eval()
            Baseline._resnet18_model = model

        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255

        if len(mask_uint8.shape) == 2:
            mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = mask_uint8

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(mask_rgb)
        input_batch = input_tensor.unsqueeze(0).to(Baseline._resnet18_device)

        with torch.no_grad():
            if layer == 'avgpool':
                x = input_batch
                x = Baseline._resnet18_model.conv1(x)
                x = Baseline._resnet18_model.bn1(x)
                x = Baseline._resnet18_model.relu(x)
                x = Baseline._resnet18_model.maxpool(x)
                x = Baseline._resnet18_model.layer1(x)
                x = Baseline._resnet18_model.layer2(x)
                x = Baseline._resnet18_model.layer3(x)
                x = Baseline._resnet18_model.layer4(x)
                x = Baseline._resnet18_model.avgpool(x)
                features = torch.flatten(x, 1)
            elif layer == 'fc':
                features = Baseline._resnet18_model(input_batch)
            else:
                raise ValueError(f"Unknown layer: {layer}. Choose 'avgpool' or 'fc'.")

        features_np = features.cpu().numpy().flatten()

        if len(features_np) == encode_len:
            encoding = features_np
        elif len(features_np) > encode_len:
            indices = np.linspace(0, len(features_np)-1, encode_len, dtype=int)
            encoding = features_np[indices]
        else:
            x_old = np.arange(len(features_np))
            x_new = np.linspace(0, len(features_np)-1, encode_len)
            interp_func = interp1d(x_old, features_np, kind='linear')
            encoding = interp_func(x_new)

        return encoding.tolist()

    @staticmethod
    def vit_shape_encoding(binary_mask, encode_len: int = 768, model_name: str = 'vit_b_16') -> list:
        """
        Extract shape embedding from a binary mask using pre-trained Vision Transformer (ViT).

        Parameters
        ----------
        binary_mask : np.ndarray, shape (H, W)
            Binary shape mask (0s and 1s, or boolean).
        encode_len : int, optional
            Desired length of the output encoding vector. Default is 768.
        model_name : str, optional
            ViT variant: 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32'.

        Returns
        -------
        encoding : list, shape (encode_len,)
            Feature embedding extracted from ViT, resized to encode_len.
        """
        if Baseline._vit_model is None or not hasattr(Baseline._vit_model, '_model_name') or Baseline._vit_model._model_name != model_name:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Baseline._vit_device = device

            if model_name == 'vit_b_16':
                model = models.vit_b_16(pretrained=True)
            elif model_name == 'vit_b_32':
                model = models.vit_b_32(pretrained=True)
            elif model_name == 'vit_l_16':
                model = models.vit_l_16(pretrained=True)
            elif model_name == 'vit_l_32':
                model = models.vit_l_32(pretrained=True)
            else:
                raise ValueError(f"Unknown model_name: {model_name}.")

            model = model.to(device)
            model.eval()
            model._model_name = model_name
            Baseline._vit_model = model

        mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255

        if len(mask_uint8.shape) == 2:
            mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = mask_uint8

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(mask_rgb)
        input_batch = input_tensor.unsqueeze(0).to(Baseline._vit_device)

        with torch.no_grad():
            x = input_batch
            x = Baseline._vit_model.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)

            batch_size = x.shape[0]
            cls_token = Baseline._vit_model.class_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

            x = x + Baseline._vit_model.encoder.pos_embedding
            x = Baseline._vit_model.encoder.dropout(x)
            x = Baseline._vit_model.encoder.layers(x)
            x = Baseline._vit_model.encoder.ln(x)

            features = x[:, 0]

        features_np = features.cpu().numpy().flatten()

        if len(features_np) == encode_len:
            encoding = features_np
        elif len(features_np) > encode_len:
            indices = np.linspace(0, len(features_np)-1, encode_len, dtype=int)
            encoding = features_np[indices]
        else:
            x_old = np.arange(len(features_np))
            x_new = np.linspace(0, len(features_np)-1, encode_len)
            interp_func = interp1d(x_old, features_np, kind='linear')
            encoding = interp_func(x_new)

        return encoding.tolist()

    @staticmethod
    def clip_encoding(mask: np.ndarray, encode_len: int, model_name: str = "ViT-B/32") -> list:
        """
        Extract latent features from a binary mask using the CLIP vision encoder.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W)
            Binary mask with values 0 (background) and 1 (foreground).
        encode_len : int
            Desired length of the output encoding vector.
        model_name : str, optional
            CLIP model variant. Default is "ViT-B/32".

        Returns
        -------
        encoding : list, shape (encode_len,)
            CLIP visual features extracted from the mask image, resized to encode_len.
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP is not installed. Install it with: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        if Baseline._clip_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Baseline._clip_model, Baseline._clip_preprocess = clip.load(model_name, device=device)
            Baseline._clip_model.eval()

        device = next(Baseline._clip_model.parameters()).device

        mask = (mask > 0).astype(np.uint8)
        mask_rgb = np.stack([mask * 255, mask * 255, mask * 255], axis=-1).astype(np.uint8)

        from PIL import Image
        pil_image = Image.fromarray(mask_rgb)
        image_input = Baseline._clip_preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = Baseline._clip_model.encode_image(image_input)

        features_np = features.cpu().numpy().flatten().astype(np.float64)

        if len(features_np) == encode_len:
            encoding = features_np
        elif len(features_np) > encode_len:
            indices = np.linspace(0, len(features_np) - 1, encode_len, dtype=int)
            encoding = features_np[indices]
        else:
            x_old = np.arange(len(features_np))
            x_new = np.linspace(0, len(features_np) - 1, encode_len)
            interp_func = interp1d(x_old, features_np, kind='linear')
            encoding = interp_func(x_new)

        return encoding.tolist()

    @staticmethod
    def shape_context_encoding(binary_mask, encode_len: int = 512) -> list:
        """
        Compute shape context encoding for a binary mask.

        Args:
            binary_mask: numpy array of shape (H, W) with values 0 (background) and 1 (foreground)
            encode_len: Desired length of the output encoding vector

        Returns:
            list of shape (encode_len,) representing the shape context descriptor
        """
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        contour = max(contours, key=len)
        contour_points = contour.squeeze(axis=1).astype(np.float64)

        if len(contour_points) < 3:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        n_sample = min(100, len(contour_points))
        indices = np.linspace(0, len(contour_points) - 1, n_sample, dtype=int)
        sampled_points = contour_points[indices]

        best_a, best_d = encode_len, 1
        for a in range(2, encode_len):
            if encode_len % a == 0:
                d = encode_len // a
                if abs(a - d) < abs(best_a - best_d):
                    best_a, best_d = a, d
        n_angle_bins, n_dist_bins = best_a, best_d

        n = len(sampled_points)
        diff = sampled_points[np.newaxis, :, :] - sampled_points[:, np.newaxis, :]

        distances = np.sqrt((diff ** 2).sum(axis=2))
        angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        np.fill_diagonal(distances, 0)
        mean_dist = distances[distances > 0].mean() if (distances > 0).any() else 1.0
        norm_distances = distances / mean_dist

        log_distances = np.log(norm_distances + 1e-10)

        valid_log_dists = log_distances[distances > 0]
        if len(valid_log_dists) == 0:
            return np.zeros(encode_len, dtype=np.float64).tolist()

        dist_min = valid_log_dists.min()
        dist_max = valid_log_dists.max()
        dist_edges = np.linspace(dist_min, dist_max + 1e-10, n_dist_bins + 1)
        angle_edges = np.linspace(0, 2 * np.pi, n_angle_bins + 1)

        mean_histogram = np.zeros((n_angle_bins, n_dist_bins), dtype=np.float64)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            pt_angles = angles[i, mask]
            pt_log_dists = log_distances[i, mask]

            hist, _, _ = np.histogram2d(
                pt_angles, pt_log_dists,
                bins=[angle_edges, dist_edges]
            )

            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum

            mean_histogram += hist

        mean_histogram /= n

        encoding = mean_histogram.flatten()
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding /= norm

        return encoding.tolist()
