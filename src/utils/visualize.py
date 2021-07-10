import numpy as np
import matplotlib.pyplot as plt


def discrete_cmap(n, base_cmap='nipy_spectral'):
    """
      Create an N-bin discrete colormap from the specified input map
      """

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


def plot_vrp(ax, instance, solution=None, node_size=100, font_size=12):
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    coords = np.array([instance.depot] + instance.customers)
    ax.scatter(instance.depot[0], instance.depot[1], c='black', s=node_size, marker='s')
    ax.text(instance.depot[0], instance.depot[1], "D", fontsize=font_size)
    if solution is not None:
        complete_routes = solution.complete_routes()
        incomplete_routes = solution.incomplete_routes()
        n_routes = len(complete_routes) + len(incomplete_routes)
        cmap = discrete_cmap(n_routes + 2)
        qvs = []
        for vehicle, route in enumerate(complete_routes):
            color = cmap(n_routes - vehicle)
            xs = np.array([x for x, _ in coords[route]])
            ys = np.array([y for _, y in coords[route]])
            ax.scatter(xs[1:-1], ys[1:-1], color=color, s=node_size)

            # TODO: Optional
            xs = xs[1:-1]
            ys = ys[1:-1]

            qv = ax.quiver(
                xs[:-1],
                ys[:-1],
                xs[1:] - xs[:-1],
                ys[1:] - ys[:-1],
                scale_units='xy',
                angles='xy',
                scale=1,
                color=color,
                label=f'Vehicle {vehicle}'
            )
            qvs.append(qv)
            ax.legend(handles=qvs)
        for i, route in enumerate(incomplete_routes):
            color = cmap(i + 1)
            route = [r for r in route if r != 0]
            xs = np.array([x for x, _ in coords[route]])
            ys = np.array([y for _, y in coords[route]])
            ax.scatter(xs, ys, color=color, s=node_size)
            ax.plot(xs, ys, '--', color=color)
        ax.set_title(solution.cost())
    else:
        for i in range(instance.n_customers):
            ax.scatter(instance.customers[i][0], instance.customers[i][1], c='orange', s=node_size)

    for i in range(instance.n_customers):
        # ax.text(instance.customers[i][0], instance.customers[i][1], str(instance.demands[i]), fontsize=font_size)
        ax.text(instance.customers[i][0], instance.customers[i][1], str(i + 1), fontsize=font_size)