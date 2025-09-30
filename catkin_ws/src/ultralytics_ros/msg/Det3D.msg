# float32 x
# float32 y
# float32 z
# float32 yaw
# float32 radius

# The 3D position and orientation of the bounding box center
geometry_msgs/Pose center

# The size of the bounding box, in meters, surrounding the object's center
#   pose.
geometry_msgs/Vector3 size

float32 score
# int8 class_id
string class_name
int32 tracked_id