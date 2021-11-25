from manim import *
import pandas as pd

# df = pd.read_csv('animation_data.csv')
# list_of_values = list(df['0'].values)[1:]
# length = len(list_of_values)
# x = range(1,length + 1)
# coords = list(zip(list_of_values,x))

# class FollowingGraphCamera(MovingCameraScene):
#     def construct(self):
#         self.camera.frame.save_state()

#         dots = VGroup(*[Dot().move_to(self.coords_to_point(coord[0],coord[1])) for coord in coords])
#         self.add(dots)

#         # create the axes and the curve
#         ax = Axes(x_range=[-1, 100], y_range=[-1, 10])
#         graph = ax.plot(list_of_values, color=BLUE)#, x_range=[0, 100])

#         # create dots based on the graph
#         moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
#         dot_1 = Dot(ax.i2gp(graph.t_min, graph))
#         dot_2 = Dot(ax.i2gp(graph.t_max, graph))

#         self.add(ax, graph, dot_1, dot_2, moving_dot)
#         self.play(self.camera.frame.animate.scale(0.5).move_to(moving_dot))

#         def update_curve(mob):
#             mob.move_to(moving_dot.get_center())

#         self.camera.frame.add_updater(update_curve)
#         self.play(MoveAlongPath(moving_dot, graph, rate_func=linear))
#         self.camera.frame.remove_updater(update_curve)

#         self.play(Restore(self.camera.frame))

class FollowingGraphCamera(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        # create the axes and the curve
        ax = Axes(x_range=[-15, 15], y_range=[-15, 15])
        graph = ax.plot(lambda x: 5 * np.sin(x), color=BLUE, x_range=[0, 6 * PI])

        # create dots based on the graph
        moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
        dot_1 = Dot(ax.i2gp(graph.t_min, graph))
        dot_2 = Dot(ax.i2gp(graph.t_max, graph))


        ellipse_1 = Ellipse(width=9.0, height=11.0, color=BLUE)
        dot = Dot()
        dott = dot.copy()
        dot2 = Dot(color=BLUE, radius=0.3, point=[4.5,0,0])
        dot4 = Dot(color=YELLOW,radius=1).shift(UP).shift(UP)

        line = Line(dot2.get_center(), dot4.get_center()).set_color(ORANGE)


        self.add(ax, graph, dot_1, dot_2, moving_dot, ellipse_1, dott, dot, line)  #,b1,b1text,pl_text,star_text)
        self.wait(1)
        self.play(self.camera.frame.animate.scale(1.5))


        self.play(GrowFromCenter(ellipse_1))
        self.play(Transform(dot, dot4))
        self.play(Transform(dott,dot2))
        self.play(MoveAlongPath(dot2, ellipse_1), run_time=10, rate_func=linear)
        self.play(line.animate(rate_func=linear))
        self.wait()
