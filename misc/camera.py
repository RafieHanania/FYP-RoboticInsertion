import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# Enable the streams you need
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Depth intrinsics
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
print("Depth:", depth_intrinsics)

# Color (RGB) intrinsics
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()
print("Color:", color_intrinsics)

# Extrinsics (depth -> color)
depth_to_color = depth_profile.get_extrinsics_to(color_profile)
print("Extrinsics:", depth_to_color)

pipeline.stop()
# ```

# The output will look something like:
# ```
# width: 640, height: 480, ppx: 321.6, ppy: 241.6, fx: 385.1, fy: 385.1, 
# model: Brown Conrady, coeffs: [0, 0, 0, 0, 0]