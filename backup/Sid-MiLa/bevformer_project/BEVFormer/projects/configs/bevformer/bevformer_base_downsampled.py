_base_ = ['./bevformer_base.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (464, 800),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}
print("Config successfully generated: Image inputs restricted to 464x800 resolution.")
