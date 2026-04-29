import numpy as np
import sys
sys.path.append("utils/")
import ShapePoseEncoder
import ShapeGeometryEncoder

class ShapeGeometryPoseEncoder:
    def __init__(self, n_max=20, res=300, lam=0.6, encode_len=128, pose_vec_len = 5, seed_sigma=0.1,
                  beta = 1):
        """pose_over_geo_weight: weight to balance pose and geometry encoding
           1: equal weight
           >1: more weight on pose
           <1: more weight on geometry
        """
        assert n_max >= 0, "n_max must be non-negative"
        self.encode_len = encode_len
        self.pose_vec_len = pose_vec_len
        self.res = res
        self.seed_sigma = seed_sigma
        self.rho = np.linspace(0, 1, self.res)
        self.theta = np.linspace(0, 2 * np.pi, self.res)
        self.r, self.t = np.meshgrid(self.rho, self.theta)
        if beta is not None:
            if beta >= 0 and beta <= 2:
                self.beta = beta
            elif abs(beta) > abs(beta - 2):
                self.beta = 2
            else:
                self.beta = 0
        else:
            self.beta = 1

        self.res = res
        self.lam = lam

        self.geometry_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(n_max=n_max, res=res, lam=lam, encode_len=encode_len)

        self.construct_pose_encoders()
    
    def construct_pose_encoders(self):
        band_indices = self.geometry_encoder.zernike_corpus.get_indices()
        self.pose_encoders = list()
        for _, m_p in band_indices:
            radial_band_num = 0
            for n, m in band_indices:
                if m == m_p:
                    radial_band_num += 1
            assert radial_band_num > 0

            pose_encoder = ShapePoseEncoder.ShapePoseEncoder(res=self.res, 
                                                            K=self.pose_vec_len, 
                                                            L=radial_band_num, 
                                                            m_p=m_p, 
                                                            seed_sigma=self.seed_sigma)
            
            self.pose_encoders.append({'m_p': m_p, 'pose_encoder': pose_encoder})
            
    def encode(self, shape_mask, pose_vec, run_freqprop=False, mask_in_euclidean=False):
        # the shape_mask is expected to be in polar coordinates, with shape (len(rho), len(theta))
        geometry_coeffs_complex = self.geometry_encoder.encode(shape_mask, mask_in_euclidean=mask_in_euclidean, return_raw_coeffs=True)[2]
        pose_encoded_rst = list()
        for pose_encoder_dict in self.pose_encoders:
            pose_encoder = pose_encoder_dict['pose_encoder']
            m_p = pose_encoder_dict['m_p']

            pose_encde_coeffs = pose_encoder.encode(pose_vec, just_return_encode=True ) 
            pose_encoded_rst.append((m_p, pose_encde_coeffs))

        for (m_p, pose_encde_coeffs) in pose_encoded_rst:
            added_id = 0
            for i, (n, m) in enumerate(self.geometry_encoder.zernike_corpus.get_indices()):
                if m == m_p:
                    if self.beta == 1:
                        geometry_coeffs_complex[i] = geometry_coeffs_complex[i]*np.exp(-1j*pose_encde_coeffs[added_id])
                    elif self.beta < 1: #emphasize pose
                        geometry_coeffs_complex[i] = np.exp(self.beta*np.log(geometry_coeffs_complex[i]))*np.exp(-1j*pose_encde_coeffs[added_id])
                    else: #emphasize geometry
                        eta_beta = np.exp(-5*(self.beta - 1))
                        geometry_coeffs_complex[i] = geometry_coeffs_complex[i]*np.exp(-1j*eta_beta*pose_encde_coeffs[added_id])
                    added_id += 1
        #run FreqProp
        if run_freqprop:
            rFreqProp_coeff = self.geometry_encoder.zernike_corpus.radial_freqprop(geometry_coeffs_complex, lam=self.lam)
            arFreqProp_coeff = self.geometry_encoder.zernike_corpus.angular_freqprop(rFreqProp_coeff, lam=self.lam)
        else:
            arFreqProp_coeff = geometry_coeffs_complex

        real_coeffs = self.geometry_encoder.get_real_compact_coeffs(arFreqProp_coeff)

        return real_coeffs