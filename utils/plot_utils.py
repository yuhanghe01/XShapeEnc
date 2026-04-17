import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def mask_to_rgb(mask, color_hex="#0078D4", bg_color=(1, 1, 1)):
    """Convert binary mask to RGB image with given mask and background color."""
    color_rgb = np.array(mcolors.to_rgb(color_hex))
    bg_rgb = np.array(bg_color)
    rgb = np.zeros((*mask.shape, 3))
    if color_hex == "#0078D4":
        rgb[mask > 0.5] = color_rgb
    else:
        alpha = 0.8
        rgb[mask > 0.5] = alpha * color_rgb + (1 - alpha) * bg_rgb
    # rgb[mask > 0.5] = color_rgb
    rgb[mask <= 0.5] = bg_rgb

    return rgb

def plot_masks(mask_list, titles=None, save_path=None):
    """
    Plot multiple masks side by side.

    Args:
        mask_list (list): list of 2D numpy arrays (masks)
        titles (list, optional): list of titles corresponding to each mask
        save_path (str, optional): path to save the figure; if None, display it
    """
    num_masks = len(mask_list)
    plt.figure(figsize=(5 * num_masks, 5))

    for i, mask in enumerate(mask_list):
        plt.subplot(1, num_masks, i + 1)
        plt.imshow(mask)
        # if titles and i < len(titles):
        #     plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mask(mask, title=None, save_path=None):
    plt.imshow(mask)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_one_shape(geom, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    if not geom.is_empty:
        minx, miny, maxx, maxy = geom.bounds
        padding = 10  # optional padding
        ax.set_xlim(minx - padding, maxx + padding)
        ax.set_ylim(miny - padding, maxy + padding)
    else:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    ax.set_aspect('equal')
    ax.axis('off')

    def plot_polygon(polygon):
        x, y = polygon.exterior.xy
        ax.fill(x, y, color = "#0078D4", edgecolor='none', alpha=0.8)
        for hole in polygon.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color='white')

    def plot_recursive(g):
        if g.is_empty:
            return
        if g.geom_type == 'Polygon':
            plot_polygon(g)
        elif g.geom_type == 'MultiPolygon':
            for poly in g.geoms:
                plot_polygon(poly)
        elif g.geom_type == 'LineString':
            x, y = g.xy
            ax.plot(x, y, color='black', linewidth=1)
        elif g.geom_type == 'Point':
            ax.plot(g.x, g.y, 'o', color='red')
        elif g.geom_type == 'MultiPoint':
            for pt in g.geoms:
                ax.plot(pt.x, pt.y, 'o', color='red')
        elif g.geom_type == 'GeometryCollection':
            for sub_geom in g.geoms:
                plot_recursive(sub_geom)

    plot_recursive(geom)

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# Visualization utility using matplotlib patches
def plot_shapes_geoms(geoms, titles=None, cols=4, figsize=(12, 8),
                      save_path="shapes.pdf"):
    microsoft_blue = "#0078D4"
    vue_green = "#42b983" #"#42b983"
    rows = (len(geoms) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, geom in enumerate(geoms):
        ax = axes[i]
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_aspect('equal')
        ax.axis('off')

        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, color = microsoft_blue, edgecolor='none', alpha=0.8)
            for holes in geom.interiors:
                hx, hy = holes.xy
                ax.fill(hx, hy, color='white')
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color = microsoft_blue, edgecolor='none', alpha=0.8)
                for holes in poly.interiors:
                    hx, hy = holes.xy
                    ax.fill(hx, hy, color='white')
        if titles:
            ax.set_title(titles[i])
        
    for i in range(len(geoms), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def plot_triptych_cartesian(zr, mask, recon_128, recon_256, recon_512, name, out_png):
    """
    Plot mask/reconstructions on true Cartesian coordinates derived from zr.r, zr.t.
    Use a light sky-blue colormap similar to Shapely's default polygon mask plots.
    """
    x = zr.r * np.cos(zr.t)
    y = zr.r * np.sin(zr.t)

    # Define a simple 2-color map: white for 0, light sky blue for >0
    cmap = mcolors.ListedColormap(["white", "deepskyblue"])

    def show(ax, Z, title):
        pcm = ax.pcolormesh(
            x, y, Z,
            shading="nearest",
            cmap=cmap,
            vmin=0.0, vmax=1.0   # force binary-like contrast
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        ax.axis("off")
        return pcm

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    show(axes[0], mask,       f"{name}: mask")
    show(axes[1], recon_128,  "recon from 128")
    show(axes[2], recon_256,  "recon from 256")
    show(axes[3], recon_512,  "recon from 512")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)