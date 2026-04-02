import pyrealsense2 as rs

ctx = rs.context()
dev = ctx.query_devices()[0]
print(f"Device: {dev.get_info(rs.camera_info.name)}\n")

seen = set()
for sensor in dev.query_sensors():
    for p in sensor.get_stream_profiles():
        if p.stream_type() == rs.stream.color:
            vp = p.as_video_stream_profile()
            key = (vp.width(), vp.height(), vp.fps(), vp.format())
            if key not in seen:
                seen.add(key)
                intr = vp.get_intrinsics()
                print(f"{vp.width()}x{vp.height()} @ {vp.fps()}fps ({vp.format()})")
                print(f"  fx={intr.fx:.2f}  fy={intr.fy:.2f}  ppx={intr.ppx:.2f}  ppy={intr.ppy:.2f}")
                print(f"  distortion: {intr.model}  coeffs: {intr.coeffs}\n")