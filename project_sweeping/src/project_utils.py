import numpy as np
from omni.isaac.core.utils.numpy.rotations import gf_quat_to_tensor, wxyz2xyzw, xyzw2wxyz
from omni.isaac.core.utils.numpy.tensor import create_zeros_tensor
from pxr import Gf
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
import yaml


def tf_matrices_from_pose(translation: np.ndarray, orientation: np.ndarray, device=None) -> np.ndarray:
    """[summary]

    Args:
        translations (Union[np.ndarray, torch.Tensor]): translations with shape (N, 3).
        orientations (Union[np.ndarray, torch.Tensor]): quaternion representation (scalar first) with shape (N, 4).

    Returns:
        Union[np.ndarray, torch.Tensor]: transformation matrices with shape (N, 4, 4)
    """
    
    result = np.zeros([4, 4], dtype=np.float32)
    r = Rotation.from_quat(orientation[[1, 2, 3, 0]])
    result[:3, :3] = r.as_matrix()
    result[:3, 3] = translation
    result[ 3, 3] = 1
    return result


def get_local_from_world(parent_transform, position, orientation, device=None):
    
    calculated_translation = create_zeros_tensor(shape=3, dtype="float32", device=device)
    calculated_orientation = create_zeros_tensor(shape=4, dtype="float32", device=device)
    my_world_transform = tf_matrices_from_pose(translation=position, orientation=orientation)
    # TODO: vectorize this
    
    robot_to_object = np.dot(np.linalg.inv(parent_transform), my_world_transform)
    
    transform = Gf.Transform()
    transform.SetMatrix(Gf.Matrix4d(np.transpose(robot_to_object).tolist()))
    calculated_translation = np.array(transform.GetTranslation())
    calculated_orientation = gf_quat_to_tensor(transform.GetRotation().GetQuat())
    
    return calculated_translation, calculated_orientation

def load_yaml_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Pose 데이터를 numpy 배열로 변환하고 정렬하는 함수
def load_and_reshape_pose(pose_dict):
    """
    Sort by (x, y) coordinates, and reshape into (1, rows, cols, 7).
    """

    # 2. Sort poses by (x, y) values
    sorted_poses = sorted(pose_dict.items(), key=lambda item: (-item[1][0], item[1][1]))

    # 3. Convert to numpy array
    pose_list = [np.array(pose, dtype=np.float32) for _, pose in sorted_poses]
    pose_array = np.array(pose_list)  # (num_objects, 7) shape

    # 4. Determine shape dynamically
    num_objects = pose_array.shape[0]  # Total number of objects
    num_rows = len(set([pose[0] for pose in pose_array]))  # Unique x values
    num_cols = num_objects // num_rows  # Compute number of columns

    # 5. Reshape into (1, rows, cols, 7)
    pose_array = pose_array.reshape(1, num_rows, num_cols, 7)

    return tuple(map(tuple, pose_array.tolist()))