import numpy as np
from kornia.geometry import Se3
from pycolmap import Rigid3d, Rotation3d


def Se3_to_Rigid3d(Se3_pose: Se3) -> Rigid3d:
    R_np = Se3_pose.rotation.matrix().numpy(force=True).astype(np.float64)
    t_np = Se3_pose.t.numpy(force=True).astype(np.float64)
    return Rigid3d(Rotation3d(R_np), t_np)
