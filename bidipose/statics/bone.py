"""Static bone definitions."""

from bidipose.statics.joints import (
    h36m_joints_name_to_index,
)

h36m_bones = [
    [h36m_joints_name_to_index["PELVIS"], h36m_joints_name_to_index["R_HIP"]],
    [h36m_joints_name_to_index["PELVIS"], h36m_joints_name_to_index["L_HIP"]],
    [h36m_joints_name_to_index["R_HIP"], h36m_joints_name_to_index["R_KNEE"]],
    [h36m_joints_name_to_index["L_HIP"], h36m_joints_name_to_index["L_KNEE"]],
    [h36m_joints_name_to_index["R_KNEE"], h36m_joints_name_to_index["R_ANKLE"]],
    [h36m_joints_name_to_index["L_KNEE"], h36m_joints_name_to_index["L_ANKLE"]],
    [h36m_joints_name_to_index["PELVIS"], h36m_joints_name_to_index["SPINE"]],
    [h36m_joints_name_to_index["SPINE"], h36m_joints_name_to_index["THORAX"]],
    [h36m_joints_name_to_index["THORAX"], h36m_joints_name_to_index["NECK"]],
    [h36m_joints_name_to_index["NECK"], h36m_joints_name_to_index["HEAD"]],
    [h36m_joints_name_to_index["THORAX"], h36m_joints_name_to_index["R_SHOULDER"]],
    [h36m_joints_name_to_index["THORAX"], h36m_joints_name_to_index["L_SHOULDER"]],
    [h36m_joints_name_to_index["R_SHOULDER"], h36m_joints_name_to_index["R_ELBOW"]],
    [h36m_joints_name_to_index["L_SHOULDER"], h36m_joints_name_to_index["L_ELBOW"]],
    [h36m_joints_name_to_index["R_ELBOW"], h36m_joints_name_to_index["R_WRIST"]],
    [h36m_joints_name_to_index["L_ELBOW"], h36m_joints_name_to_index["L_WRIST"]],
]
