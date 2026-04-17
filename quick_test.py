import numpy as np
import ShapePoseEncoder
import ShapeGeometryEncoder
import ShapeGeometryPoseEncoder

def get_circle_mask(res=300):
    rho = np.linspace(0, 1, res)
    theta = np.linspace(0, 2 * np.pi, res)
    r, t = np.meshgrid(rho, theta)

    # Convert polar to cartesian
    x = r * np.cos(t)
    y = r * np.sin(t)

    # Define circle within unit disk
    circle_mask = (x**2 + y**2) <= 1.

    return circle_mask

def get_square_mask(res=300):
    # get one square mask in polar coordinates
    rho = np.linspace(0, 1, res)
    theta = np.linspace(0, 2 * np.pi, res)
    r, t = np.meshgrid(rho, theta)

    # Convert polar to cartesian
    x = r * np.cos(t)
    y = r * np.sin(t)
    # Define square within unit disk
    # Side length sqrt(2), i.e. half length = 1/√2 ~ 0.707 to fit in unit disk
    half_len = 1 / np.sqrt(2)
    square_mask = (np.abs(x) <= half_len) & (np.abs(y) <= half_len)

    return square_mask

def test_pose_encoder():
    pos_vec = np.array([0.2, 0.5, 0.7, 0.9, 0.4]).tolist()
    encode_len = 128
    encoder = ShapePoseEncoder.ShapePoseEncoder(res=300, K=len(pos_vec), L=encode_len, m_p=17, seed_sigma=0.1)
    _, coeffs, _, p_recovered = encoder.encode(pos_vec)
    print("Original Pose Vector:", pos_vec)
    print("Recovered Pose Vector:", p_recovered)
    print("Recovery Error:", np.linalg.norm(pos_vec - p_recovered))
    print("Zernike Coefficients:", coeffs)


def test_geometry_encoder():
    geometry_encoder = ShapeGeometryEncoder.ShapeGeometryEncoder(n_max=5, res=300, lam=0.6, encode_len=512)
    square_mask = get_square_mask(res=300)
    circle_mask = get_circle_mask(res=300)

    square_encoded = geometry_encoder.encode(square_mask.astype(np.float64))
    circle_encoded = geometry_encoder.encode(circle_mask.astype(np.float64))

    square_recon = geometry_encoder.decode(square_encoded)
    circle_recon = geometry_encoder.decode(circle_encoded)

    MSE_square = np.mean((square_mask.astype(np.float64) - square_recon)**2)
    MSE_circle = np.mean((circle_mask.astype(np.float64) - circle_recon)**2)

    print("Square Mask MSE:", MSE_square)
    print("Circle Mask MSE:", MSE_circle)

def test_geometry_pose_encoder():
    shape_geometry_pose_encoder = ShapeGeometryPoseEncoder.ShapeGeometryPoseEncoder(n_max=20, res=300, lam=0.6, encode_len=512, pose_vec_len=5, beta=1)
    square_mask = get_square_mask(res=300)
    pose_vec = np.array([0.2, 0.5, 0.7, 0.9, 0.4]).tolist()

    geo_pose_encoding = shape_geometry_pose_encoder.encode(square_mask.astype(np.float64),
                                                        pose_vec=pose_vec,
                                                        mask_in_euclidean=False,)
    
    print("Geometry-Pose Encoding:", geo_pose_encoding)




if __name__ == "__main__":
    test_pose_encoder()
    test_geometry_encoder()
    test_geometry_pose_encoder()