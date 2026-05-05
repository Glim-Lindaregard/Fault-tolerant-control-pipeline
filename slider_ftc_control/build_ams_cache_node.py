import pickle
from pathlib import Path

import rclpy
from rclpy.node import Node

from slider_ftc_control.config import make_default_config
from slider_ftc_control.ams import build_ams_cache
from slider_ftc_control.ams.simplified_cache import build_simplified_ams_cache


class BuildAMSCacheNode(Node):
    def __init__(self):
        super().__init__("build_ams_cache_node")

        self.declare_parameter(
            "cache_path",
            str(Path.home() / "slider_ws" / "ams_cache.pkl")
        )

        cache_path = Path(self.get_parameter("cache_path").value)

        self.get_logger().info("Building AMS cache...")
        cfg = make_default_config()

        ams_cache = build_ams_cache(cfg)
        simplified_ams_cache = build_simplified_ams_cache(cfg, ams_cache)

        cache = {
            "ams_cache": ams_cache,
            "simplified_ams_cache": simplified_ams_cache,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

        self.get_logger().info(f"Saved AMS cache to {cache_path}")


def main(args=None):
    rclpy.init(args=args)
    node = BuildAMSCacheNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()