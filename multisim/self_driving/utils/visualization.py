import os
from typing import List, Tuple

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
from shapely.affinity import translate, rotate
from descartes import PolygonPatch
from math import atan2, degrees


# https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread
from self_driving.road import Road


class RoadTestVisualizer:
    """
        Visualize and Plot RoadTests
    """

    little_triangle = Polygon([(2, 0), (0, -1), (0, 1), (2, 0)])
    square = Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1), (1,1)])
    
    ax = None
    
    def __init__(self, map_size, ax = None):
        self.map_size = map_size
        self.last_submitted_test_figure = None
        self.ax = ax
        # Make sure there's a windows and does not block anything when calling show
        # plt.ion()
        # plt.show()

    def _setup_figure(self) -> None:
        if self.last_submitted_test_figure is not None:
            # Make sure we operate on the right figure
            plt.figure(self.last_submitted_test_figure.number)
            plt.clf()
            self.ax = self.ax
        else:
            self.last_submitted_test_figure = plt.figure()

        if self.ax is None:
            self.ax = plt.gca()

        # plt.gcf().set_title("Last Generated Test")
        self.ax.set_aspect('equal', 'box')
        self.ax.set(xlim=(-30, self.map_size + 30), ylim=(-30, self.map_size + 30))

    def visualize_road_test(
            self,
            road: Road,
            folder_path: str = os.getcwd(),
            filename: str = 'road_0',
            invert: bool = False,
            car_trajectory: List[Tuple[float]] = None,
            plot_control_points: bool = False,
            do_save: bool = True
    ) -> None:

        self._setup_figure()

        # plt.draw()
        # plt.pause(0.001)
        
        # Plot the map. Trying to re-use an artist in more than one Axes which is supported
        map_patch = patches.Rectangle((0, 0), self.map_size, self.map_size, linewidth=1, edgecolor='black', facecolor='none')
        self.ax.add_patch(map_patch)

        # Road Geometry.
        if not invert:
            road_poly = LineString([(t[0], t[1]) for t in road.get_concrete_representation(to_plot=True)]).buffer(8.0, cap_style=2, join_style=2)
        else:
            road_poly = LineString([(t[0], t[1]) for t in road.get_inverse_concrete_representation(to_plot=True)]).buffer(8.0, cap_style=2, join_style=2)

        if car_trajectory is not None or plot_control_points:
            # blur the road such that the trajectory of the car is visible on the road
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray', alpha=0.4)  # ec='#555555', alpha=0.5, zorder=4)
        else:
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)

        self.ax.add_patch(road_patch)

        # Interpolated Points
        if not invert:
            sx = [t[0] for t in road.get_concrete_representation(to_plot=True)]
            sy = [t[1] for t in road.get_concrete_representation(to_plot=True)]
        else:
            sx = [t[0] for t in road.get_inverse_concrete_representation(to_plot=True)]
            sy = [t[1] for t in road.get_inverse_concrete_representation(to_plot=True)]

        plt.plot(sx, sy, 'yellow')

        # Plot the little triangle indicating the starting position of the ego-vehicle
        delta_x = sx[1] - sx[0]
        delta_y = sy[1] - sy[0]

        current_angle = atan2(delta_y, delta_x)

        rotation_angle = degrees(current_angle)
        transformed_fov = rotate(self.little_triangle, origin=(0, 0), angle=rotation_angle)
        transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
        plt.plot(*transformed_fov.exterior.xy, color='black')

        # Plot the little square indicating the ending position of the ego-vehicle
        delta_x = sx[-1] - sx[-2]
        delta_y = sy[-1] - sy[-2]

        current_angle = atan2(delta_y, delta_x)

        rotation_angle = degrees(current_angle)
        transformed_fov = rotate(self.square, origin=(0, 0), angle=rotation_angle)
        transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
        plt.plot(*transformed_fov.exterior.xy, color='black')

        plt.draw()

        if car_trajectory is not None:
            car_trajectory_x = [cr_item[0] for cr_item in car_trajectory]
            car_trajectory_y = [cr_item[1] for cr_item in car_trajectory]
            plt.scatter(car_trajectory_x, car_trajectory_y, color="red")

        if plot_control_points:
            control_points_xs = [cp.x for cp in road.control_points]
            control_points_ys = [cp.y for cp in road.control_points]
            print(control_points_xs)
            print(control_points_ys)
            plt.scatter(control_points_xs, control_points_ys, color="red", marker="*", s=50)

        plt.pause(0.001)

        if do_save:
            plt.savefig(os.path.join(folder_path, '{}.png'.format(filename)))
            plt.close()

    def visualize_road_basic(
            self,
            road_points: List[Tuple],
            width: int = 4,
            folder_path: str = os.getcwd(),
            filename: str = 'road_0',
            car_trajectory: List[Tuple[float]] = None,
            do_save: bool = True,
            invert: bool = False,
            plot_control_points: bool = False,
            do_show_plot: bool = False,
            simulator: str = None,
            road = None, # for control points retrieval
            format = "png",
            no_center = False
    ) -> None:

        # self._setup_figure()

        # plt.draw()
        # plt.pause(0.001)
        
        # Plot the map. Trying to re-use an artist in more than one Axes which is supported
        # map_patch = patches.Rectangle((0, 0), self.map_size, self.map_size, linewidth=1, edgecolor='black', facecolor='none')
        # self.ax.add_patch(map_patch)

        # apply displacement for beamng roads
        offset_x = 0
        offset_y = 0
        if simulator.lower() == "beamng":
            road_points = [(p[0] - offset_x, p[1] - offset_y) for p in road_points]
            width = width * 2 # need to resize as only both directions of lane are visualized together

        # Road Geometry.
        if not invert:
            road_poly = LineString([(t[0], t[1]) for t in road_points]).buffer( int(width)/2, cap_style=2, join_style=2)
        # else:
        #     road_poly = LineString([(t[0], t[1]) for t in road_points]).buffer(8.0, cap_style=2, join_style=2)
        if car_trajectory is not None or plot_control_points:
            # blur the road such that the trajectory of the car is visible on the road
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray', alpha=0.3)  # ec='#555555', alpha=0.5, zorder=4)
        else:
            road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)

        self.ax.add_patch(road_patch)

        # Interpolated Points
        if not invert:
            sx = [t[0] for t in road_points]
            sy = [t[1] for t in road_points]
        # else:
        #     sx = [t[0] for t in road.get_inverse_concrete_representation(to_plot=True)]
        #     sy = [t[1] for t in road.get_inverse_concrete_representation(to_plot=True)]

        if not no_center:
            self.ax.plot(sx, sy, 'yellow')

        # Plot the little triangle indicating the starting position of the ego-vehicle
        delta_x = sx[1] - sx[0]
        delta_y = sy[1] - sy[0]

        current_angle = atan2(delta_y, delta_x)

        rotation_angle = degrees(current_angle)
        transformed_fov = rotate(self.little_triangle, origin=(0, 0), angle=rotation_angle)
        transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
        self.ax.plot(*transformed_fov.exterior.xy, color='black')

        # # Plot the little square indicating the ending position of the ego-vehicle
        # delta_x = sx[-1] - sx[-2]
        # delta_y = sy[-1] - sy[-2]

        # current_angle = atan2(delta_y, delta_x)

        # rotation_angle = degrees(current_angle)
        # transformed_fov = rotate(self.square, origin=(0, 0), angle=rotation_angle)
        # transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
        # plt.plot(*transformed_fov.exterior.xy, color='black')

        # plt.draw()

        if car_trajectory is not None:
            car_trajectory_x = [cr_item[0] for cr_item in car_trajectory]
            car_trajectory_y = [cr_item[1] for cr_item in car_trajectory]
            self.ax.scatter(car_trajectory_x, car_trajectory_y, color="red", s=1)

        if road is not None and plot_control_points:
            control_points_xs = [cp.x for cp in road.control_points]
            control_points_ys = [cp.y for cp in road.control_points]
            print(control_points_xs)
            print(control_points_ys)
            plt.scatter(control_points_xs, control_points_ys, color="black", marker="*", s=50, zorder = 10)
        
        if do_save:
            plt.savefig(os.path.join(folder_path, '{}.{}'.format(filename, format)),format=format)
            plt.clf()
            plt.close()
        if do_show_plot:
            plt.show()

if __name__ == "__main__":
    import config
    # from self_driving.utils.example_bng import road_2 as road, car_trajectory_2 as car_trajectory
    from self_driving.utils.example_bng_dnk_2 import road as road, bng_trajectory as car_trajectory

    points=[(p[0],p[1]) for p in road ]

    offset_x = abs(car_trajectory[0][0] - points[0][0])
    offset_y = abs(car_trajectory[0][1] - points[0][1])

    width = road[0][3]

    points = [(p[0] - offset_x, p[1]) for p in points]

    plt.plot([p[0] for p in points], 
             [p[1] for p in points], 
             linewidth=5, 
             c="#808080")

    # plot lanes, border
    from self_driving.utils.visualization import RoadTestVisualizer
    f = plt.figure()
    ax = plt.gca()
    plt.axis('equal')

    road_test_visualizer = RoadTestVisualizer(map_size=config.MAP_SIZE, ax = ax)
    road_test_visualizer.visualize_road_basic(road_points=points, 
                                    folder_path= os.getcwd() + "/self_driving/utils/",
                                    width=4,
                                    do_save=False,
                                    do_show_plot=False,
                                    car_trajectory=car_trajectory,
                                    filename="bng_driving_2")
    f.tight_layout()
    ##############################

    from self_driving.utils.example_bng_dnk_2 import road as road, dnk_trajectory as car_trajectory

    points=[(p[0],p[1]) for p in road ]
    width = road[0][3]
    plt.plot([p[0] for p in points], 
             [p[1] for p in points], 
             linewidth=5, 
             c="#808080")
    # plot lanes, border
    from self_driving.utils.visualization import RoadTestVisualizer
    f = plt.figure()
    ax = plt.gca()
    plt.axis('equal')

    road_test_visualizer = RoadTestVisualizer(map_size=config.MAP_SIZE, ax = ax)
    road_test_visualizer.visualize_road_basic(road_points=points, 
                                    folder_path= os.getcwd() + "/self_driving/utils/",
                                    width=width / 2,
                                    do_save=True,
                                    car_trajectory=car_trajectory,
                                    filename="dnk_driving_2")
    f.tight_layout()

    # ###################

    # import config
    # # from self_driving.utils.example_bng import road_2 as road, car_trajectory_2 as car_trajectory
    # from self_driving.utils.example_bng_dnk_1 import road as road, bng_trajectory as car_trajectory

    # points=[(p[0],p[1]) for p in road ]
 
    # offset_x = abs(car_trajectory[0][0] - points[0][0])
    # offset_y = abs(car_trajectory[0][1] - points[0][1])

    # width = road[0][3]

    # points = [(p[0] - offset_x, p[1] + offset_y) for p in points]

    # plt.plot([p[0] for p in points], 
    #          [p[1] for p in points], 
    #          linewidth=5, 
    #          c="#808080")
    # # plot lanes, border
    # from self_driving.utils.visualization import RoadTestVisualizer
    # f = plt.figure()
    # ax = plt.gca()
    # plt.axis('equal')

    # road_test_visualizer = RoadTestVisualizer(map_size=config.MAP_SIZE, ax = ax)
    # road_test_visualizer.visualize_road_basic(road_points=points, 
    #                                 folder_path= os.getcwd() + "/self_driving/utils/",
    #                                 width=width/2,
    #                                 do_save=True,
    #                                 car_trajectory=car_trajectory,
    #                                 filename="bng_driving_1")
    # f.tight_layout()
    
    # from self_driving.utils.example_bng_dnk_1 import road as road, dnk_trajectory as car_trajectory

    # points=[(p[0],p[1]) for p in road ]
    # width = road[0][3]
    # plt.plot([p[0] for p in points], 
    #          [p[1] for p in points], 
    #          linewidth=5, 
    #          c="#808080")
    # # plot lanes, border
    # from self_driving.utils.visualization import RoadTestVisualizer
    # f = plt.figure()
    # ax = plt.gca()
    # plt.axis('equal')

    # road_test_visualizer = RoadTestVisualizer(map_size=config.MAP_SIZE, ax = ax)
    # road_test_visualizer.visualize_road_basic(road_points=points, 
    #                                 folder_path= os.getcwd() + "/self_driving/utils/",
    #                                 width=width / 2,
    #                                 do_save=True,
    #                                 car_trajectory=car_trajectory,
    #                                 filename="dnk_driving_1")
    # f.tight_layout()







