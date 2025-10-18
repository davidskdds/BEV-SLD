import os
import struct
import numpy as np
import shutil
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_ros1
from utils import save_pcd_open3d, get_config


def read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
    """Pure-Python PointCloud2 reader compatible with sensor_msgs/PointCloud2."""
    fmt = []
    offsets = {}
    for field in msg.fields:
        if field.name in field_names:
            datatype_to_struct = {
                1: ('b', 1),  # INT8
                2: ('B', 1),  # UINT8
                3: ('h', 2),  # INT16
                4: ('H', 2),  # UINT16
                5: ('i', 4),  # INT32
                6: ('I', 4),  # UINT32
                7: ('f', 4),  # FLOAT32
                8: ('d', 8),  # FLOAT64
            }
            ctype, size = datatype_to_struct[field.datatype]
            fmt.append((field.name, field.offset, ctype))
            offsets[field.name] = (field.offset, size)
    step = msg.point_step
    unpack_str = "<" + "".join([c for _, _, c in fmt])
    record_size = struct.calcsize(unpack_str)

    for i in range(0, len(msg.data), step):
        record = msg.data[i : i + record_size]
        vals = struct.unpack(unpack_str, record)
        if skip_nans and any(np.isnan(v) for v in vals):
            continue
        yield vals


def main():
    cfg = get_config()
    shutil.rmtree(cfg.pc_dir, ignore_errors=True)
    os.makedirs(cfg.pc_dir, exist_ok=True)

    iteration = -1
    use_every_nth = 1

    with Reader(cfg.bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == cfg.pc_topic_name]
        if not conns:
            raise RuntimeError(f"Topic {cfg.pc_topic_name} not found in {cfg.bag_path}")

        for conn, timestamp, raw in tqdm(reader.messages(connections=conns)):
            iteration += 1
            if iteration % use_every_nth != 0:
                continue

            msg = deserialize_ros1(raw, conn.msgtype)
            msg_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            points = np.array(
                list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
                dtype=np.float32,
            ).T  # shape (3, N)

            save_pcd_open3d(points, os.path.join(cfg.pc_dir, f"{msg_stamp_sec:.6f}.pcd"))



if __name__ == "__main__":
    main()