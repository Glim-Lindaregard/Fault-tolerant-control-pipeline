import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- USED FOR TESTING --- 
def plot_ams(U, Asys):

    U = np.asarray(U)
    Asys = np.asarray(Asys)

    n_thrusters, four, n_facets = U.shape
    assert four == 4, "U must be N x 4 x nFacets"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for k in range(n_facets):
        Uk = U[:, :, k]           # N x 4
        Vk = Asys @ Uk            # 3 x 4

        verts = Vk.T              # 4 x 3 (Fx,Fy,Tau)

        poly = Poly3DCollection(
            [verts],
            facecolors=(0.8, 0.95, 0.95, 0.6),
            edgecolors=(0.3, 0.3, 0.3, 1.0),
            linewidths=0.4,
        )
        ax.add_collection3d(poly)

    # Simple auto scaling
    pts = np.vstack([ (Asys @ U[:, :, k]).T for k in range(n_facets) ])
    xmin, ymin, zmin = pts.min(axis=0)
    xmax, ymax, zmax = pts.max(axis=0)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_xlabel("F_x")
    ax.set_ylabel("F_y")
    ax.set_zlabel("Tau")

    plt.show()


if __name__ == "__main__":
    from slider_ftc_control.config import make_default_config
    from slider_ftc_control.ams import build_ams_cache
    import numpy as np

    cfg = make_default_config()
    cache = build_ams_cache(cfg)
    mode = cache[-1]

    U = np.stack(mode.facets, axis=2)
    Asys = mode.A_mode

    plot_ams(U, Asys)
