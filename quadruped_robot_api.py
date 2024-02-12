import tkinter as tk
import tkinter.ttk as ttk
import tkinter.simpledialog as sd
import tkinter.messagebox as ms
import random
import time
import numpy as np
import cyipopt

# this class creates instances of the API/GUI of the quadruped robot
class quadruped_robot_api():
    def __init__(self, root, instance):
        self.root = root
        self.root.title(f"Quadruped robot api {instance+1}")
        self.root.geometry("+0+0")
        self.root.resizable(False, False)
        self.instance = instance
        self.grid_color_theme = 0  # 0 = blue/green, 1 = red/yellow
        self.grid_color_themes_list = ["blue/\ngreen", "red/\nyellow"]  # list of possible color themes of the grid
        # define the workspace parameters
        self.x_axis_range = 5  # the maximum range from the origin to the positive x axis of the workspace
        self.y_axis_range = 5  # the maximum range from the origin to the positive y axis of the workspace
        self.z_axis_range = 5  # the maximum range from the origin to the positive z axis of the workspace
        self.axis_range_values = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]  # the possible values of the axis ranges of the workspace
        self.magnify_workspace_constant = 60  # the constant used to magnify the workspace
        self.sensitivity_values = [0.1, 0.5, 1.0]; self.sensitivity_degrees = ["high", "normal", "low"]  # the possible values/degrees of the sensitivity of the workspace control
        self.workspace_sensitivity = self.sensitivity_values[self.sensitivity_degrees.index("normal")]  # control the workspace movement sensitivity
        self.axis_terrain_enable = "on"  # control the axis and terrain visualization
        self.quadruped_robot_enable = "on"  # control the quadruped robot visualization (points and edges)
        self.quadruped_points_enable = "on"  # control the quadruped robot points visualization
        self.pointing_to_point = "(0.00, 0.00, 0.00)"  # the point to which the user's cursor is pointing
        self.axis_terrain_points_num = 8  # the number of the axis and terrain points of the workspace
        self.feet_number = 4  # the number of feet of the quadruped robot
        self.quadruped_robot_points_num = 2 * self.feet_number + 8 + 1  # the number of the points of the quadruped robot in the workspace
        self.total_points_num = self.axis_terrain_points_num + self.quadruped_robot_points_num  # the total number of the points of the workspace
        self.axis_terrain_points = np.zeros((self.axis_terrain_points_num, 3), dtype = float)  # initialize the points of the workspace
        self.quadruped_robot_points = np.zeros((self.quadruped_robot_points_num, 3), dtype = float)  # initialize the points of the quadruped robot
        self.transformed_quadruped_robot_points = np.copy(self.quadruped_robot_points)  # initialize the transformed points of the quadruped robot
        self.x_transfer_quadruped_com = 0; self.y_transfer_quadruped_com = 0; self.z_transfer_quadruped_com = 0  # the transfer of the quadruped robot's center of mass in the workspace
        self.rotate_quadruped_matrix = np.eye(4)  # the matrix used to rotate the quadruped robot in the workspace
        self.simulation_speed_values = [0.1, 0.5, 1., 2., 5.]; self.simulation_speed_degrees = ["very slow", "slow", "normal", "fast", "very fast"]  # the possible values/degrees of the simulation speed
        self.simulation_speed = self.simulation_speed_values[self.simulation_speed_degrees.index("normal")]  # control the simulation speed
        # define the quadruped robot model technical features
        self.default_mass = 30.4213964625  # the default mass of the quadruped robot in kg
        self.default_g = 9.81  # the default gravitational acceleration in m/s^2
        self.default_I = np.array([[0.88201174, -0.00137526, -0.00062895], [-0.00137526, 1.85452968, -0.00018922], [-0.00062895, -0.00018922, 1.97309185]])  # the default inertia tensor of the quadruped robot in kg*m^2
        self.default_feet_pos = np.array([[0.34, 0.19, -0.42],\
                                    [-0.34, 0.19, -0.42],\
                                    [0.34, -0.19, -0.42],\
                                    [-0.34, -0.19, -0.42]])  # the default relative positions (in m) of left fore foot, left hind foot, right fore foot and right hind foot respectively
        self.default_feet_x_dist = abs(self.default_feet_pos[0][0] - self.default_feet_pos[1][0])  # the default distance in x axis between the left fore foot and the left hind foot in m
        self.default_feet_y_dist = abs(self.default_feet_pos[0][1] - self.default_feet_pos[2][1])  # the default distance in y axis between the left fore foot and the right fore foot in m
        self.default_feet_height = 4/5 * abs(self.default_feet_pos[0][2])  # the default height of the quadruped robot in m (the default height of the center of mass of the quadruped robot)
        self.default_body_length_x = self.default_feet_x_dist * 3/2  # the x length of the quadruped body in m
        self.default_body_length_y = self.default_feet_y_dist * 3/2  # the y length of the quadruped body in m
        self.default_body_length_z = self.default_feet_height / 2  # the z length of the quadruped body in m
        self.mass = self.default_mass  # the mass of the quadruped robot in kg
        self.mass_bounds = [0.1, 1000]  # the bounds of the mass of the quadruped robot in kg
        self.g = self.default_g  # the gravitational acceleration in m/s^2
        self.gravity_bounds = [0.1, 100]  # the bounds of the gravitational acceleration in m/s^2
        self.I = np.copy(self.default_I)  # the inertia tensor of the quadruped robot in kg*m^2
        self.I_components_bounds = [-np.inf, np.inf]  # the bounds of the inertia tensor components of the quadruped robot in kg*m^2
        self.feet_pos = np.copy(self.default_feet_pos)  # the relative positions of left fore foot, left hind foot, right fore foot and right hind foot in m
        self.feet_pos_bounds = [-self.axis_range_values[-1], self.axis_range_values[-1]]  # the bounds of the position of the left fore foot of the quadruped robot in m
        self.feet_height = self.default_feet_height  # the height of the quadruped robot in m (the height of the center of mass of the quadruped robot)
        self.feet_height_bounds = [0.05, 2.0]  # the bounds of the feet height of the quadruped robot in m
        self.feet_x_dist = self.default_feet_x_dist  # the distance in x axis between the left fore foot and the left hind foot in m
        self.feet_x_dist_bounds = [0.0, 2.0]  # the bounds of the distance in x axis between the left fore foot and the left hind foot of the quadruped robot in m
        self.feet_y_dist = self.default_feet_y_dist  # the distance in y axis between the left fore foot and the right fore foot in m
        self.feet_y_dist_bounds = [0.0, 2.0]  # the bounds of the distance in y axis between the left fore foot and the right fore foot of the quadruped robot in m
        self.body_length_x = self.default_body_length_x  # the x length of the quadruped body in m
        self.body_length_y = self.default_body_length_y  # the y length of the quadruped body in m
        self.body_length_z = self.default_body_length_z  # the z length of the quadruped body in m
        self.center_of_mass = np.array([self.feet_pos[0][0] - self.feet_x_dist/2, self.feet_pos[0][1] - self.feet_y_dist/2, self.feet_pos[0][2] + self.feet_height + self.body_length_z/2])  # the position of the center of mass of the quadruped robot in m
        self.legs_bounds_x = []  # the feet/legs bounds along the x-axis
        self.legs_bounds_y = []  # the feet/legs bounds along the y-axis
        self.legs_bounds_z = []  # the feet/legs bounds along the z-axis
        for foot in range(self.feet_number):
            self.legs_bounds_x.append([self.feet_pos[foot][0] - self.body_length_x/2, self.feet_pos[foot][0] + self.body_length_x/2])  # initialization of the feet/legs bounds along the x-axis
            self.legs_bounds_y.append([self.feet_pos[foot][1] - self.body_length_y/2, self.feet_pos[foot][1] + self.body_length_y/2])  # initialization of the feet/legs bounds along the y-axis
            self.legs_bounds_z.append([-1.5*(self.feet_height + self.body_length_z/2), -0.5*(self.feet_height + self.body_length_z/2)])  # initialization of the feet/legs bounds along the z-axis
        # define the simulation parameters
        self.dt = 0.1  # the time step of the simulation in sec
        self.dt_values = [0.01, 0.02, 0.05, 0.1];  # the possible values of the time step of the simulation
        self.total_time = 2  # the total time of the simulation in sec
        self.total_time_values = [0.5, 1, 2, 3, 4, 5, 10, 15, 20]  # the possible values of the total time of the simulation
        self.cycles_period = 1  # the period of every cycle in sec
        self.cycles_period_values = [0.5, 1, 2, 3, 4, 5, 10]  # the possible values of the cycles period of the simulation
        self.cycles_number = int(self.total_time / self.cycles_period)  # the number of cycles of the simulation
        self.gaits_period = 0.1  # the period of every gait in sec
        self.gaits_period_values = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5]  # the possible values of the gaits period of the simulation
        self.gaits_number = int(self.total_time / self.gaits_period)  # the number of gaits of the simulation
        self.initial_com_position = np.array([1, 1, self.feet_height + self.body_length_z/2])  # the initial position of the quadruped robot's center of mass in m
        self.initial_body_orientation = np.array([0, 0, 0])  # the initial rotation of the quadruped robot's body, ZYX Euler angles in degrees
        self.final_com_position = np.array([2.5, 2, self.feet_height + self.body_length_z/2])  # the final position of the quadruped robot's center of mass in m
        self.final_body_orientation = np.array([45, 0, 0])  # the final rotation of the quadruped robot's body, ZYX Euler angles in degrees
        self.simulation_is_running = False  # the flag that indicates if the simulation is running
        # define the gaits sequence variables
        self.gaits_sequence = []  # the gaits sequence for all the feet of the quadruped robot
        self.current_total_time = self.total_time  # the current total time of the simulation
        self.current_cycles_period = self.cycles_period  # the current number of cycles of the gaits sequence/simulation
        self.current_gaits_period = self.gaits_period  # the current number of gaits of the gaits sequence/simulation
        self.chosen_cycle_tens = 0  # the tens of the chosen cycle number
        self.chosen_cycle_units = 0  # the units of the choesn cycle number
        self.chosen_move_type = "walk"  # the movement type of the quadruped robot (walk, trot, pace, jump) at the chosen cycle
        self.move_types_list = ["walk", "trot", "pace", "run", "jump", "all C", "all S"]  # the list of the possible movement types of the quadruped robot
        self.move_types_contact_phases = [[[[0.0, 0.2], [0.3, 0.7], [0.8, 1.0]], [[0.0, 0.1], [0.2, 0.6], [0.7, 1.0]], [[0.1, 0.5], [0.6, 1.0]], [[0.0, 0.4], [0.5, 0.9]]],\
                                          [[[0.0, 0.1], [0.4, 1.0]], [[0.0, 0.6], [0.9, 1.0]], [[0.0, 0.6], [0.9, 1.0]], [[0.0, 0.1], [0.4, 1.0]]],\
                                          [[[0.0, 0.6], [0.9, 1.0]], [[0.0, 0.6], [0.9, 1.0]], [[0.0, 0.1], [0.4, 1.0]], [[0.0, 0.1], [0.4, 1.0]]],\
                                          [[[0.5, 0.9]], [[0.0, 0.4]], [[0.5, 0.9]], [[0.0, 0.4]]],\
                                          [[[0.0, 0.4], [0.8, 1.0]], [[0.0, 0.4], [0.8, 1.0]], [[0.0, 0.4], [0.8, 1.0]], [[0.0, 0.4], [0.8, 1.0]]],\
                                          [[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]],\
                                          [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]]  # the list of the contact phases (feet sequences) for every movement type
        # define the variables for the non linear trajectory optimization/planning
        self.body_state_dim = 13  # the number of the body state variables
        self.body_com_dim = 6  # the number of the body center of mass variables (position and velocity)
        self.body_position_dim = 3  # the number of the body position variables
        self.feet_state_dim = 3 * self.feet_number  # the number of the feet state variables
        self.N = self.body_state_dim + self.feet_state_dim  # the total number of the state variables
        self.M = 3 * self.feet_number  # the total number of the control input variables
        self.K = 0  # the total_number of trajectory knot points
        self.trajectory_states_list = []  # the list of the trajectory state variables
        self.trajectory_control_inputs_list = []  # the list of the trajectory control input variables
        self.quadruped_traj_com_locations = []  # the list of the quadruped robot's center of mass locations along the calculated trajectory
        self.quadruped_traj_body_orientations = []  # the list of the quadruped robot's body orientations along the calculated trajectory
        self.quadruped_traj_feet_positions = []  # the list of the quadruped robot's feet positions along the calculated trajectory
        self.trajectory_steps_counter = 0  # the counter of the trajectory steps
        # the initial actions for the workspace (where the quadruped robot operates)
        self.create_workspace_menus_options()  # create the workspace and the menus with the options given to the user
        self.switch_coor_system_matrix = np.array([[0, 1, 0, self.workspace_width/2], [0, 0, -1, self.workspace_height/2], [1, 0, 0, 0], [0, 0, 0, 1]])  # transformation matrix needed because of the difference between workspace and canvas coordinates systems
        self.create_workspace_points_links()  # create the points of the visualized workspace
        self.reset_workspace()  # reset the position of the axis origin to be on the center of the workspace canvas
        self.calculate_draw_new_quadruped_model()  # calculate and draw the quadruped robot model in the workspace
        self.draw_next_workspace_frame()  # begin the loop for controlling the motion of the workspace visualization

    def create_workspace_menus_options(self):  # create the workspace and the menus with the options given to the user
        # create the workspace (where the quadruped robot operates)
        self.workspace_height = 4/5 * self.root.winfo_screenheight(); self.workspace_width = 0.85*1/2 * self.root.winfo_screenwidth()
        self.workspace = tk.Canvas(self.root, width = self.workspace_width, height = self.workspace_height, bg = "yellow")
        self.workspace.grid(row = 1, rowspan = 2, column = 1, sticky = tk.NSEW)
        # create the binds for the quadruped robot workspace visualization
        self.workspace.bind("<Button-3>", lambda event: self.transfer_workspace_start(event))
        self.workspace.bind("<B3-Motion>", lambda event: self.transfer_workspace(event))
        self.workspace.bind("<Double-Button-3>", lambda event: self.reset_workspace(event))
        self.workspace.bind("<Button-1>", lambda event: self.rotate_workspace_start(event))
        self.workspace.bind("<B1-Motion>", lambda event: self.rotate_workspace(event))
        self.workspace.bind("<MouseWheel>", lambda event: self.scale_workspace(event))
        for point in range(self.total_points_num):  # bind the points of the workspace to show their coordinates when the user's cursor is pointing to them
            self.workspace.tag_unbind(f"point{point}", "<Enter>"); self.workspace.tag_bind(f"point{point}", "<Enter>", self.show_point_coordinates_helper(point))
        # create the borders and controls of the workspace
        borders_width = 50
        workspace_edges_color = "cyan"
        self.workspace_up_edge = tk.Frame(self.root, width = 2*self.workspace_width, height = borders_width, bg = workspace_edges_color)
        self.workspace_left_edge = tk.Frame(self.root, width = 2*borders_width, height = self.workspace_height, bg = workspace_edges_color)
        self.workspace_right_edge = tk.Frame(self.root, width = borders_width, height = self.workspace_height, bg = workspace_edges_color)
        self.workspace_up_edge.grid(row = 0, column = 0, columnspan = 5, sticky = tk.NSEW)
        self.workspace_left_edge.grid(row = 1, rowspan = 2, column = 0, sticky = tk.NSEW)
        self.workspace_right_edge.grid(row = 1, rowspan = 2, column = 2, sticky = tk.NSEW)
        API_title_ord = 1; API_title_x = 1/2; menu_label(self.workspace_up_edge, "Quadruped Robot API:", f"Calibri 20 bold", "black", workspace_edges_color, API_title_x * 2*(self.workspace_width+borders_width), API_title_ord * borders_width / 2)
        left_edge_width = 2*borders_width; left_edge_height = self.workspace_height; left_edge_rows = 16; left_edge_font = 12
        sensitivity_label_ord = 1; sensitivity_label_x = 1/2; menu_label(self.workspace_left_edge, "Control\nsensitivity:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, sensitivity_label_x * left_edge_width, sensitivity_label_ord * left_edge_height / (left_edge_rows + 1))
        sensitivity_button_ord = sensitivity_label_ord+1; sensitivity_button_x = 1/2; self.change_control_sensitivity_button = menu_button(self.workspace_left_edge, self.sensitivity_degrees[1], f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, sensitivity_button_x * left_edge_width, sensitivity_button_ord * left_edge_height / (left_edge_rows + 1), self.change_workspace_control_sensitivity).button
        x_axis_range_label_ord = 3; x_axis_range_label_x = 1/2; menu_label(self.workspace_left_edge, "X axis range:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, x_axis_range_label_x * left_edge_width, x_axis_range_label_ord * left_edge_height / (left_edge_rows + 1))
        x_axis_range_button_ord = x_axis_range_label_ord+1; x_axis_range_button_x = 1/2; self.change_x_axis_range_button = menu_button(self.workspace_left_edge, f"{self.x_axis_range}", f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, x_axis_range_button_x * left_edge_width, x_axis_range_button_ord * left_edge_height / (left_edge_rows + 1), self.change_x_axis_range).button
        y_axis_range_label_ord = 5; y_axis_range_label_x = 1/2; menu_label(self.workspace_left_edge, "Y axis range:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, y_axis_range_label_x * left_edge_width, y_axis_range_label_ord * left_edge_height / (left_edge_rows + 1))
        y_axis_range_button_ord = y_axis_range_label_ord+1; y_axis_range_button_x = 1/2; self.change_y_axis_range_button = menu_button(self.workspace_left_edge, f"{self.y_axis_range}", f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, y_axis_range_button_x * left_edge_width, y_axis_range_button_ord * left_edge_height / (left_edge_rows + 1), self.change_y_axis_range).button
        z_axis_range_label_ord = 7; z_axis_range_label_x = 1/2; menu_label(self.workspace_left_edge, "Z axis range:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, z_axis_range_label_x * left_edge_width, z_axis_range_label_ord * left_edge_height / (left_edge_rows + 1))
        z_axis_range_button_ord = z_axis_range_label_ord+1; z_axis_range_button_x = 1/2; self.change_z_axis_range_button = menu_button(self.workspace_left_edge, f"{self.z_axis_range}", f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, z_axis_range_button_x * left_edge_width, z_axis_range_button_ord * left_edge_height / (left_edge_rows + 1), self.change_z_axis_range).button
        show_axis_terrain_label_ord = 9; show_axis_terrain_label_x = 1/2; menu_label(self.workspace_left_edge, "Axis/\nTerrain:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, show_axis_terrain_label_x * left_edge_width, show_axis_terrain_label_ord * left_edge_height / (left_edge_rows + 1))
        show_axis_terrain_button_ord = show_axis_terrain_label_ord+1; show_axis_terrain_button_x = 1/2; self.show_axis_terrain_button = menu_button(self.workspace_left_edge, self.axis_terrain_enable, f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, show_axis_terrain_button_x * left_edge_width, show_axis_terrain_button_ord * left_edge_height / (left_edge_rows + 1), self.show_axis_terrain).button
        show_quadruped_robot_label_ord = 11; show_quadruped_robot_label_x = 1/2; menu_label(self.workspace_left_edge, "Quadruped\nrobot:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, show_quadruped_robot_label_x * left_edge_width, show_quadruped_robot_label_ord * left_edge_height / (left_edge_rows + 1))
        show_quadruped_robot_button_ord = show_quadruped_robot_label_ord+1; show_quadruped_robot_button_x = 1/2; self.show_quadruped_robot_button = menu_button(self.workspace_left_edge, self.quadruped_robot_enable, f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, show_quadruped_robot_button_x * left_edge_width, show_quadruped_robot_button_ord * left_edge_height / (left_edge_rows + 1), self.show_quadruped_robot).button
        show_quadruped_points_label_ord = 13; show_quadruped_points_label_x = 1/2; menu_label(self.workspace_left_edge, "Quadruped\npoints:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, show_quadruped_points_label_x * left_edge_width, show_quadruped_points_label_ord * left_edge_height / (left_edge_rows + 1))
        show_quadruped_points_button_ord = show_quadruped_points_label_ord+1; show_quadruped_points_button_x = 1/2; self.show_quadruped_points_button = menu_button(self.workspace_left_edge, self.quadruped_points_enable, f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, show_quadruped_points_button_x * left_edge_width, show_quadruped_points_button_ord * left_edge_height / (left_edge_rows + 1), self.show_quadruped_points).button
        simulation_speed_label_ord = 15; simulation_speed_label_x = 1/2; menu_label(self.workspace_left_edge, "Simulation\nspeed:", f"Calibri {left_edge_font} bold", "black", workspace_edges_color, simulation_speed_label_x * left_edge_width, simulation_speed_label_ord * left_edge_height / (left_edge_rows + 1))
        simulation_speed_button_ord = simulation_speed_label_ord+1; simulation_speed_button_x = 1/2; self.change_simulation_speed_button = menu_button(self.workspace_left_edge, f"normal", f"Calibri {left_edge_font} bold", "magenta", workspace_edges_color, simulation_speed_button_x * left_edge_width, simulation_speed_button_ord * left_edge_height / (left_edge_rows + 1), self.change_simulation_speed).button
        # create the menus' backgrounds
        menus_background_width = self.workspace_width / 2; menus_background_height = self.workspace_height - 5 * borders_width
        self.menu1_width = menus_background_width; self.menu1_height = menus_background_height; self.menu1_rows = 9; menu1_font = 12; menu1_bg_color = "black"
        self.menu2_width = menus_background_width; self.menu2_height = menus_background_height; self.menu2_rows = 10; menu2_font = 12; menu2_bg_color = "black"
        self.menu3_width = 2*menus_background_width; self.menu3_height = self.workspace_height - menus_background_height; self.menu3_rows = 7; menu3_font = 12; menu3_bg_color = "black"
        self.menu1 = tk.Frame(self.root, width = self.menu1_width, height = self.menu1_height, bg = menu1_bg_color, highlightbackground = "red", highlightcolor = "red", highlightthickness = 5)
        self.menu1.grid(row = 1, column = 3, sticky = tk.NSEW)
        self.menu2 = tk.Frame(self.root, width = self.menu2_width, height = self.menu2_height, bg = menu2_bg_color, highlightbackground = "red", highlightcolor = "red", highlightthickness = 5)
        self.menu2.grid(row = 1, column = 4, sticky = tk.NSEW)
        self.menu3 = tk.Frame(self.root, width = self.menu3_width, height = self.menu3_height, bg = menu3_bg_color, highlightbackground = "red", highlightcolor = "red", highlightthickness = 5)
        self.menu3.grid(row = 2, column = 3, columnspan = 2, sticky = tk.NSEW)
        # create the options of menu 1
        title1_ord = 1; title1_x = 1/2; menu_label(self.menu1, "Quadruped robot model:", f"Arial {menu1_font} bold underline", "gold", menu1_bg_color, title1_x * self.menu1_width, title1_ord * self.menu1_height / (self.menu1_rows + 1))
        mass_ord = 2; mass_label_x = 1/5; menu_label(self.menu1, "Mass m\n(kg):", f"Arial {menu1_font} bold", "lime", menu1_bg_color, mass_label_x * self.menu1_width, mass_ord * self.menu1_height / (self.menu1_rows + 1))
        mass_button_x = 2/5; self.change_mass_button = menu_button(self.menu1, "m", f"Calibri {menu1_font} bold", "white", menu1_bg_color, mass_button_x * self.menu1_width, mass_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_mass).button
        gravity_ord = 2; gravity_label_x = 3/5; menu_label(self.menu1, "Gravity g\n(m/s^2):", f"Arial {menu1_font} bold", "lime", menu1_bg_color, gravity_label_x * self.menu1_width, gravity_ord * self.menu1_height / (self.menu1_rows + 1))
        gravity_button_x = 4/5; self.change_gravity_button = menu_button(self.menu1, "g", f"Calibri {menu1_font} bold", "white", menu1_bg_color, gravity_button_x * self.menu1_width, gravity_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_gravity).button
        I_ord = 3.5; I_label_x = 1/5; menu_label(self.menu1, "Inertia\ntensor I\n(kg*m^2):", f"Arial {menu1_font} bold", "lime", menu1_bg_color, I_label_x * self.menu1_width, I_ord * self.menu1_height / (self.menu1_rows + 1))
        Ixx_ord = I_ord-0.5; Ixx_x = 2/5; self.change_Ixx_button = menu_button(self.menu1, "Ixx", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Ixx_x * self.menu1_width, Ixx_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Ixx_inertia).button
        Iyy_ord = I_ord-0.5; Iyy_x = 3/5; self.change_Iyy_button = menu_button(self.menu1, "Iyy", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Iyy_x * self.menu1_width, Iyy_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Iyy_inertia).button
        Izz_ord = I_ord-0.5; Izz_x = 4/5; self.change_Izz_button = menu_button(self.menu1, "Izz", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Izz_x * self.menu1_width, Izz_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Izz_inertia).button
        Ixy_ord = I_ord+0.5; Ixy_x = 2/5; self.change_Ixy_button = menu_button(self.menu1, "Ixy", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Ixy_x * self.menu1_width, Ixy_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Ixy_inertia).button
        Ixz_ord = I_ord+0.5; Ixz_x = 3/5; self.change_Ixz_button = menu_button(self.menu1, "Ixz", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Ixz_x * self.menu1_width, Ixz_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Ixz_inertia).button
        Iyz_ord = I_ord+0.5; Iyz_x = 4/5; self.change_Iyz_button = menu_button(self.menu1, "Iyz", f"Calibri {menu1_font} bold", "white", menu1_bg_color, Iyz_x * self.menu1_width, Iyz_ord * self.menu1_height / (self.menu1_rows + 1), self.change_quadruped_Iyz_inertia).button
        body_details_label_ord = 5.5; body_details_label_x = 1/4; menu_label(self.menu1, "Body\ndetails\n(m):", f"Arial {menu1_font} bold", "lime", menu1_bg_color, body_details_label_x * self.menu1_width, body_details_label_ord * self.menu1_height / (self.menu1_rows + 1))
        body_length_x_ord = body_details_label_ord-0.5; body_length_x_button_x = 2/4; self.change_body_length_x_button = menu_button(self.menu1, "length x", f"Calibri {menu1_font} bold", "white", menu1_bg_color, body_length_x_button_x * self.menu1_width, body_length_x_ord * self.menu1_height / (self.menu1_rows + 1), self.change_body_length_x).button
        body_length_y_ord = body_details_label_ord-0.5; body_length_y_button_x = 3/4; self.change_body_length_y_button = menu_button(self.menu1, "length y", f"Calibri {menu1_font} bold", "white", menu1_bg_color, body_length_y_button_x * self.menu1_width, body_length_y_ord * self.menu1_height / (self.menu1_rows + 1), self.change_body_length_y).button
        body_length_z_ord = body_details_label_ord+0.5; body_length_z_button_x = 2/4; self.change_body_length_z_button = menu_button(self.menu1, "length z", f"Calibri {menu1_font} bold", "white", menu1_bg_color, body_length_z_button_x * self.menu1_width, body_length_z_ord * self.menu1_height / (self.menu1_rows + 1), self.change_body_length_z).button
        adjust_inertia_ord = body_details_label_ord+0.5; adjust_inertia_button_x = 3/4; self.adjust_inertia_button = menu_button(self.menu1, "adjust I", f"Calibri {menu1_font} bold", "white", menu1_bg_color, adjust_inertia_button_x * self.menu1_width, adjust_inertia_ord * self.menu1_height / (self.menu1_rows + 1), self.adjust_quadruped_inertia).button
        feet_details_label_ord = 7.5; feet_details_label_x = 1/4; menu_label(self.menu1, "Feet\ndetails\n(m):", f"Arial {menu1_font} bold", "lime", menu1_bg_color, feet_details_label_x * self.menu1_width, feet_details_label_ord * self.menu1_height / (self.menu1_rows + 1))
        left_fore_foot_pos_ord = feet_details_label_ord-0.5; left_fore_foot_pos_button_x = 2/4; self.change_left_fore_foot_button = menu_button(self.menu1, "LF pos", f"Calibri {menu1_font} bold", "white", menu1_bg_color, left_fore_foot_pos_button_x * self.menu1_width, left_fore_foot_pos_ord * self.menu1_height / (self.menu1_rows + 1), self.change_left_fore_foot_position).button
        feet_height_ord = feet_details_label_ord-0.5; feet_height_button_x = 3/4; self.change_feet_height_button = menu_button(self.menu1, "height", f"Calibri {menu1_font} bold", "white", menu1_bg_color, feet_height_button_x * self.menu1_width, feet_height_ord * self.menu1_height / (self.menu1_rows + 1), self.change_feet_height).button
        left_hind_foot_dist_ord = feet_details_label_ord+0.5; left_hind_foot_dist_button_x = 2/4; self.change_left_hind_foot_dist_button = menu_button(self.menu1, "LF -> LH", f"Calibri {menu1_font} bold", "white", menu1_bg_color, left_hind_foot_dist_button_x * self.menu1_width, left_hind_foot_dist_ord * self.menu1_height / (self.menu1_rows + 1), self.change_dist_from_left_hind_foot).button
        right_fore_foot_dist_ord = feet_details_label_ord+0.5; right_fore_foot_dist_button_x = 3/4; self.change_right_fore_foot_dist_button = menu_button(self.menu1, "LF -> RF", f"Calibri {menu1_font} bold", "white", menu1_bg_color, right_fore_foot_dist_button_x * self.menu1_width, right_fore_foot_dist_ord * self.menu1_height / (self.menu1_rows + 1), self.change_dist_from_right_fore_foot).button
        model_label_ord = 9; model_label_x = 1/4; menu_label(self.menu1, "Model:", f"Arial {menu1_font} bold", "lime", menu1_bg_color, model_label_x * self.menu1_width, model_label_ord * self.menu1_height / (self.menu1_rows + 1))
        show_model_ord = 9; show_model_x = 2/4; self.show_model_button = menu_button(self.menu1, "show", f"Calibri {menu1_font} bold", "white", menu1_bg_color, show_model_x * self.menu1_width, show_model_ord * self.menu1_height / (self.menu1_rows + 1), self.show_current_quadruped_robot_model).button
        default_model_ord = 9; default_model_x = 3/4; self.default_model_button = menu_button(self.menu1, "default", f"Calibri {menu1_font} bold", "white", menu1_bg_color, default_model_x * self.menu1_width, default_model_ord * self.menu1_height / (self.menu1_rows + 1), self.get_default_quadruped_robot_model).button
        # create the options of menu 2
        title2_ord = 1; title2_x = 1/2; menu_label(self.menu2, "Optimization / Simulation options:", f"Arial {menu2_font} bold underline", "gold", menu2_bg_color, title2_x * self.menu1_width, title2_ord * self.menu2_height / (self.menu2_rows + 1))
        total_time_ord = 2; total_time_label_x = 1/3; menu_label(self.menu2, "Total time (sec):", f"Arial {menu2_font} bold", "lime", menu2_bg_color, total_time_label_x * self.menu1_width, total_time_ord * self.menu2_height / (self.menu2_rows + 1))
        total_time_button_x = 2/3; self.change_total_time_button = menu_button(self.menu2, self.total_time, f"Calibri {menu2_font} bold", "white", menu2_bg_color, total_time_button_x * self.menu2_width, total_time_ord * self.menu2_height / (self.menu2_rows + 1), self.change_simulation_total_time).button
        cycles_period_ord = 3; cycles_period_label_x = 1/5; menu_label(self.menu2, "Cycles\nT (sec):", f"Arial {menu2_font} bold", "lime", menu2_bg_color, cycles_period_label_x * self.menu1_width, cycles_period_ord * self.menu2_height / (self.menu2_rows + 1))
        cycles_period_button_x = 2/5; self.change_cycles_period_button = menu_button(self.menu2, self.cycles_period, f"Calibri {menu2_font} bold", "white", menu2_bg_color, cycles_period_button_x * self.menu2_width, cycles_period_ord * self.menu2_height / (self.menu2_rows + 1), self.change_simulation_cycles_period).button
        cycles_number_label_x = 3/5; menu_label(self.menu2, "Cycles\nnumber:", f"Arial {menu2_font} bold", "lime", menu2_bg_color, cycles_number_label_x * self.menu1_width, cycles_period_ord * self.menu2_height / (self.menu2_rows + 1))
        cycles_number_indicator_x = 4/5; self.cycles_number_indicator = menu_label(self.menu2, self.cycles_number, f"Calibri {menu2_font} bold", "yellow", menu2_bg_color, cycles_number_indicator_x * self.menu2_width, cycles_period_ord * self.menu2_height / (self.menu2_rows + 1)).label
        gaits_period_ord = 4.5; gaits_period_label_x = 1/5; menu_label(self.menu2, "Gaits\nT (sec):", f"Arial {menu2_font} bold", "lime", menu2_bg_color, gaits_period_label_x * self.menu1_width, gaits_period_ord * self.menu2_height / (self.menu2_rows + 1))
        gaits_period_button_x = 2/5; self.change_gaits_period_button = menu_button(self.menu2, self.gaits_period, f"Calibri {menu2_font} bold", "white", menu2_bg_color, gaits_period_button_x * self.menu2_width, gaits_period_ord * self.menu2_height / (self.menu2_rows + 1), self.change_simulation_gaits_period).button
        gaits_number_label_x = 3/5; menu_label(self.menu2, "Gaits\nnumber:", f"Arial {menu2_font} bold", "lime", menu2_bg_color, gaits_number_label_x * self.menu1_width, gaits_period_ord * self.menu2_height / (self.menu2_rows + 1))
        gaits_number_indicator_x = 4/5; self.gaits_number_indicator = menu_label(self.menu2, self.gaits_number, f"Calibri {menu2_font} bold", "yellow", menu2_bg_color, gaits_number_indicator_x * self.menu2_width, gaits_period_ord * self.menu2_height / (self.menu2_rows + 1)).label
        menu2_seperator_ord = 5.5; menu2_seperator_x = 1/2; menu_label(self.menu2, "----------", f"Arial {menu2_font} bold", "brown", menu2_bg_color, menu2_seperator_x * self.menu1_width, menu2_seperator_ord * self.menu2_height / (self.menu2_rows + 1))
        dt_ord = 6; dt_label_x = 1/3; menu_label(self.menu2, "dt (sec):", f"Arial {menu2_font} bold", "lime", menu2_bg_color, dt_label_x * self.menu1_width, dt_ord * self.menu2_height / (self.menu2_rows + 1))
        dt_button_x = 2/3; self.change_dt_button = menu_button(self.menu2, self.dt, f"Calibri {menu2_font} bold", "white", menu2_bg_color, dt_button_x * self.menu2_width, dt_ord * self.menu2_height / (self.menu2_rows + 1), self.change_simulation_dt).button
        initial_state_ord = 7.2; initial_state_label_x = 1/5; menu_label(self.menu2, "Initial\nstate:", f"Arial {menu2_font} bold", "lime", menu2_bg_color, initial_state_label_x * self.menu1_width, initial_state_ord * self.menu2_height / (self.menu2_rows + 1))
        initial_com_pos_button_ord = initial_state_ord-0.4; initial_com_pos_button_x = 2/4; self.change_initial_com_pos_button = menu_button(self.menu2, "position", f"Calibri {menu2_font} bold", "white", menu2_bg_color, initial_com_pos_button_x * self.menu2_width, initial_com_pos_button_ord * self.menu2_height / (self.menu2_rows + 1), self.change_quadruped_initial_position).button
        initial_body_orient_button_ord = initial_state_ord+0.4; initial_body_orient_button_x = 2/4; self.change_initial_body_orient_button = menu_button(self.menu2, "orientation", f"Calibri {menu2_font} bold", "white", menu2_bg_color, initial_body_orient_button_x * self.menu2_width, initial_body_orient_button_ord * self.menu2_height / (self.menu2_rows + 1), self.change_quadruped_initial_orientation).button
        show_initial_state_button_ord = initial_state_ord; show_initial_state_button_x = 3/4; self.show_initial_state_button = menu_button(self.menu2, "show", f"Calibri {menu2_font} bold", "white", menu2_bg_color, show_initial_state_button_x * self.menu2_width, show_initial_state_button_ord * self.menu2_height / (self.menu2_rows + 1), self.visualize_quadruped_initial_state).button
        final_state_ord = 8.7; final_state_label_x = 1/5; menu_label(self.menu2, "Final\nstate:", f"Arial {menu2_font} bold", "lime", menu2_bg_color, final_state_label_x * self.menu1_width, final_state_ord * self.menu2_height / (self.menu2_rows + 1))
        final_com_pos_button_ord = final_state_ord-0.4; final_com_pos_button_x = 2/4; self.change_final_com_pos_button = menu_button(self.menu2, "position", f"Calibri {menu2_font} bold", "white", menu2_bg_color, final_com_pos_button_x * self.menu2_width, final_com_pos_button_ord * self.menu2_height / (self.menu2_rows + 1), self.change_quadruped_final_position).button
        final_body_orient_button_ord = final_state_ord+0.4; final_body_orient_button_x = 2/4; self.change_final_body_orient_button = menu_button(self.menu2, "orientation", f"Calibri {menu2_font} bold", "white", menu2_bg_color, final_body_orient_button_x * self.menu2_width, final_body_orient_button_ord * self.menu2_height / (self.menu2_rows + 1), self.change_quadruped_final_orientation).button
        show_final_state_button_ord = final_state_ord; show_final_state_button_x = 3/4; self.show_initial_state_button = menu_button(self.menu2, "show", f"Calibri {menu2_font} bold", "white", menu2_bg_color, show_final_state_button_x * self.menu2_width, show_final_state_button_ord * self.menu2_height / (self.menu2_rows + 1), self.visualize_quadruped_final_state).button
        optimization_simulation_label_ord = 10; optimization_simulation_label_x = 1/5; menu_label(self.menu2, "Optimization/\nSimulation:", f"Arial {menu2_font} bold", "lime", menu2_bg_color, optimization_simulation_label_x * self.menu1_width, optimization_simulation_label_ord * self.menu2_height / (self.menu2_rows + 1))
        run_optimization_simulation_button_x = 2/4; self.run_optimization_simulation_button = menu_button(self.menu2, "START", f"Calibri {menu2_font} bold", "white", menu2_bg_color, run_optimization_simulation_button_x * self.menu2_width, optimization_simulation_label_ord * self.menu2_height / (self.menu2_rows + 1), self.run_optimization_simulation).button
        show_trajectory_button_x = 3/4; self.show_quadruped_trajectory_button = menu_button(self.menu2, "show", f"Calibri {menu2_font} bold", "white", menu2_bg_color, show_trajectory_button_x * self.menu2_width, optimization_simulation_label_ord * self.menu2_height / (self.menu2_rows + 1), self.show_quadruped_trajectory).button
        # create the options of menu 3
        title3_ord = 0.5; title3_x = 1/2; menu_label(self.menu3, "Gaits Sequence / Scheduling for the feet of the quadruped robot:", f"Arial {menu3_font} bold underline", "gold", menu3_bg_color, title3_x * self.menu3_width, title3_ord * self.menu3_height / (self.menu3_rows + 1))
        left_fore_foot_label_ord = 1.5; left_fore_foot_label_x = 1/8; menu_label(self.menu3, "Left Fore (LF):", f"Arial {menu3_font} bold", "lime", menu3_bg_color, left_fore_foot_label_x * self.menu3_width, left_fore_foot_label_ord * self.menu3_height / (self.menu3_rows + 1))
        left_hind_foot_label_ord = left_fore_foot_label_ord+1; left_hind_foot_label_x = 1/8; menu_label(self.menu3, "Left Hind (LH):", f"Arial {menu3_font} bold", "lime", menu3_bg_color, left_hind_foot_label_x * self.menu3_width, left_hind_foot_label_ord * self.menu3_height / (self.menu3_rows + 1))
        right_fore_foot_label_ord = left_fore_foot_label_ord+2; right_fore_foot_label_x = 1/8; menu_label(self.menu3, "Right Fore (RF):", f"Arial {menu3_font} bold", "lime", menu3_bg_color, right_fore_foot_label_x * self.menu3_width, right_fore_foot_label_ord * self.menu3_height / (self.menu3_rows + 1))
        right_hind_foot_label_ord = left_fore_foot_label_ord+3; right_hind_foot_label_x = 1/8; menu_label(self.menu3, "Right Hind (RH):", f"Arial {menu3_font} bold", "lime", menu3_bg_color, right_hind_foot_label_x * self.menu3_width, right_hind_foot_label_ord * self.menu3_height / (self.menu3_rows + 1))
        self.gaits_sequence_background_width = 0.95*3/4 * self.menu3_width; self.gaits_sequence_background_height = 4 * self.menu3_height / (self.menu3_rows + 1)
        self.gaits_sequence_background = tk.Canvas(self.menu3, width = self.gaits_sequence_background_width, height = self.gaits_sequence_background_height, bg = "black", bd = 2, relief = "solid")
        self.gaits_sequence_background.place(x = 1/4 * self.menu3_width, y = (left_fore_foot_label_ord - 0.5) * self.menu3_height / (self.menu3_rows + 1))
        current_total_time_label_ord = 5.8; current_total_time_label_x = 1/8; self.current_total_time_label = menu_label(self.menu3, f"Total time (sec): {self.current_total_time}", f"Arial {menu3_font-3} bold", "pink", menu3_bg_color, current_total_time_label_x * self.menu3_width, current_total_time_label_ord * self.menu3_height / (self.menu3_rows + 1)).label
        current_cycles_period_label_ord = 6.5; current_cycles_period_label_x = 1/8; self.current_cycles_period_label = menu_label(self.menu3, f"Cycles period (sec): {self.current_cycles_period}", f"Arial {menu3_font-3} bold", "pink", menu3_bg_color, current_cycles_period_label_x * self.menu3_width, current_cycles_period_label_ord * self.menu3_height / (self.menu3_rows + 1)).label
        current_gaits_period_label_ord = 7.2; current_gaits_period_label_x = 1/8; self.current_gaits_period_label = menu_label(self.menu3, f"Gaits period (sec): {self.current_gaits_period}", f"Arial {menu3_font-3} bold", "pink", menu3_bg_color, current_gaits_period_label_x * self.menu3_width, current_gaits_period_label_ord * self.menu3_height / (self.menu3_rows + 1)).label
        chosen_cycle_label_ord = 6; chosen_cycle_label_x = 1.15/3; menu_label(self.menu3, "Cycle number:", f"Arial {menu3_font} bold", "lime", menu3_bg_color, chosen_cycle_label_x * self.menu3_width, chosen_cycle_label_ord * self.menu3_height / (self.menu3_rows + 1))
        chosen_cycle_tens_button_ord = chosen_cycle_label_ord; chosen_cycle_tens_button_x = chosen_cycle_label_x+(80-7)/self.menu3_width; self.change_chosen_cycle_tens_button = menu_button(self.menu3, self.chosen_cycle_tens, f"Calibri {menu3_font} bold", "white", menu3_bg_color, chosen_cycle_tens_button_x * self.menu3_width, chosen_cycle_tens_button_ord * self.menu3_height / (self.menu3_rows + 1), self.change_chosen_cycle_tens).button
        chosen_cycle_units_button_ord = chosen_cycle_label_ord; chosen_cycle_units_button_x = chosen_cycle_label_x+(80+7)/self.menu3_width; self.change_chosen_cycle_units_button = menu_button(self.menu3, self.chosen_cycle_units, f"Calibri {menu3_font} bold", "white", menu3_bg_color, chosen_cycle_units_button_x * self.menu3_width, chosen_cycle_units_button_ord * self.menu3_height / (self.menu3_rows + 1), self.change_chosen_cycle_units).button
        chosen_move_type_label_ord = chosen_cycle_label_ord+1; chosen_move_type_label_x = chosen_cycle_label_x; menu_label(self.menu3, "Move type:", f"Arial {menu3_font} bold", "lime", menu3_bg_color, chosen_move_type_label_x * self.menu3_width, chosen_move_type_label_ord * self.menu3_height / (self.menu3_rows + 1))
        chosen_move_type_button_ord = chosen_move_type_label_ord; chosen_move_type_button_x = chosen_move_type_label_x+80/self.menu3_width; self.change_chosen_move_type_button = menu_button(self.menu3, self.chosen_move_type, f"Calibri {menu3_font} bold", "white", menu3_bg_color, chosen_move_type_button_x * self.menu3_width, chosen_move_type_button_ord * self.menu3_height / (self.menu3_rows + 1), self.change_chosen_move_type).button
        apply_move_to_cycle_button_ord = chosen_cycle_label_ord; apply_move_to_cycle_button_x = chosen_move_type_label_x+180/self.menu3_width; self.apply_move_to_cycle_button = menu_button(self.menu3, "apply to cycle", f"Calibri {menu3_font} bold", "white", menu3_bg_color, apply_move_to_cycle_button_x * self.menu3_width, apply_move_to_cycle_button_ord * self.menu3_height / (self.menu3_rows + 1), self.apply_move_type_to_cycle).button
        apply_move_to_all_cycles_button_ord = chosen_cycle_label_ord+1; apply_move_to_all_cycles_button_x = chosen_move_type_label_x+180/self.menu3_width; self.apply_move_to_all_cycles_button = menu_button(self.menu3, "apply to all", f"Calibri {menu3_font} bold", "white", menu3_bg_color, apply_move_to_all_cycles_button_x * self.menu3_width, apply_move_to_all_cycles_button_ord * self.menu3_height / (self.menu3_rows + 1), self.apply_move_type_to_all_cycles).button
        make_new_grid_button_ord = 6.5; make_new_grid_button_x = 9/10; self.make_new_grid_button = menu_button(self.menu3, "new\ngrid", f"Calibri {menu3_font} bold", "white", menu3_bg_color, make_new_grid_button_x * self.menu3_width, make_new_grid_button_ord * self.menu3_height / (self.menu3_rows + 1), self.make_gaits_sequence_grid).button
        self.make_gaits_sequence_grid()
    def create_workspace_points_links(self, event = None):  # create the points and links of the workspace (axis and the quadruped robot)
        self.axis_terrain_points = np.zeros((self.axis_terrain_points_num, 3), dtype = float)  # initialize the points of the workspace
        self.quadruped_robot_points = np.zeros((self.quadruped_robot_points_num, 3), dtype = float)  # initialize the points of the quadruped robot
        # create the points of the workspace
        self.axis_terrain_points[0] = [0, 0, 0]; self.axis_terrain_points[1] = [self.x_axis_range, 0, 0]; self.axis_terrain_points[2] = [0, self.y_axis_range, 0]; self.axis_terrain_points[3] = [0, 0, self.z_axis_range]
        self.axis_terrain_points[4] = [0, 0, 0]; self.axis_terrain_points[5] = [self.x_axis_range, 0, 0]; self.axis_terrain_points[6] = [self.x_axis_range, self.y_axis_range, 0]; self.axis_terrain_points[7] = [0, self.y_axis_range, 0]
        self.axis_terrain_points = np.concatenate((self.axis_terrain_points, np.ones((self.axis_terrain_points_num, 1))), axis = 1)
        for i in range(2*self.feet_number):
            self.quadruped_robot_points[i] = self.feet_pos[i%self.feet_number]+np.array([0, 0, (i//self.feet_number)*self.feet_height])
        body_surplus_x = self.body_length_x - self.feet_x_dist; body_surplus_y = self.body_length_y - self.feet_y_dist
        for j in range(self.quadruped_robot_points_num - 2*self.feet_number-1):
            self.quadruped_robot_points[2*self.feet_number+j] = self.feet_pos[j%4] + np.array([(-1)**j*body_surplus_x/2, (-1)**(j//2)*body_surplus_y/2, self.feet_height+(j//4)*self.body_length_z])
        self.quadruped_robot_points[-1] = self.center_of_mass
        self.quadruped_robot_points = np.concatenate((self.quadruped_robot_points, np.ones((self.quadruped_robot_points_num, 1))), axis = 1)
        # create the links (connecting lines) between the points
        self.points_links = self.total_points_num * [None]
        self.points_links[0] = [1, 2, 3]
        links_counter = self.axis_terrain_points_num
        for i in range(self.feet_number):
            self.points_links[self.axis_terrain_points_num+i] = [self.axis_terrain_points_num+i + self.feet_number]
        links_counter = self.axis_terrain_points_num + 2*self.feet_number
        for j in range(4):
            if j == 0 or j == 3:
                self.points_links[links_counter+j] = [links_counter+j + 4, links_counter+j + (-1)**j, links_counter+j + 2*(-1)**j]
                self.points_links[links_counter+j + 4] = [links_counter+j + 4 + (-1)**j, links_counter+j + 4 + 2*(-1)**j]
            else:
                self.points_links[links_counter+j] = [links_counter+j + 4]

    def apply_workspace_transformation(self, event = None):  # apply the transformation defined by the proper transfer, rotation and scale variables to all the points of the workspace
        self.workspace_transfer_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, self.y_cor_workspace_center], [0, 0, 1, self.z_cor_workspace_center], [0, 0, 0, 1]])
        self.workspace_scale_matrix = np.array([[self.magnify_workspace_constant * self.scale_parameter, 0, 0, 0], [0, self.magnify_workspace_constant * self.scale_parameter, 0, 0], [0, 0, self.magnify_workspace_constant * self.scale_parameter, 0], [0, 0, 0, 1]])
        self.workspace_y_rot_matrix = np.array([[np.cos(self.rot_y_workspace * np.pi / 180), 0, np.sin(self.rot_y_workspace * np.pi / 180), 0], [0, 1, 0, 0], [-np.sin(self.rot_y_workspace * np.pi / 180), 0, np.cos(self.rot_y_workspace * np.pi / 180), 0], [0, 0, 0, 1]])
        self.workspace_z_rot_matrix = np.array([[np.cos(self.rot_z_workspace * np.pi / 180), -np.sin(self.rot_z_workspace * np.pi / 180), 0, 0], [np.sin(self.rot_z_workspace * np.pi / 180), np.cos(self.rot_z_workspace * np.pi / 180), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.workspace_rotation_matrix = self.workspace_y_rot_matrix @ self.workspace_z_rot_matrix
        self.workspace_transformation_matrix = self.workspace_transfer_matrix @ self.workspace_rotation_matrix @ self.workspace_scale_matrix
        self.apply_quadruped_robot_transformation()  # apply the transformation defined by the proper transfer and rotation matrices to the points of the quadruped robot
        self.workspace_points = np.concatenate((self.axis_terrain_points, self.transformed_quadruped_robot_points), axis = 0)  # the points of the workspace, before the workspace transformation (due to the user's mouse control) is applied 
        self.canvas_moved_points = (self.switch_coor_system_matrix @ self.workspace_transformation_matrix @ np.concatenate((self.axis_terrain_points, self.transformed_quadruped_robot_points), axis = 0).T).T  # the moved points of the workspace, converted to canvas coordinates, after the workspace transformation is applied
    def apply_quadruped_robot_transformation(self, event = None):  # apply the transformation defined by the proper transfer and rotation matrices to the points of the quadruped robot
        reset_transfer_matrix = np.array([[1, 0, 0, -self.center_of_mass[0]], [0, 1, 0, -self.center_of_mass[1]], [0, 0, 1, -self.center_of_mass[2]], [0, 0, 0, 1]])  # the reset transfer matrix
        transfer_quadruped_matrix = np.array([[1, 0, 0, self.x_transfer_quadruped_com], [0, 1, 0, self.y_transfer_quadruped_com], [0, 0, 1, self.z_transfer_quadruped_com], [0, 0, 0, 1]])  # the transfer matrix of the quadruped robot
        rotate_quadruped_matrix = np.concatenate((self.rotate_quadruped_matrix, np.array([[0, 0, 0]]).T), axis = 1); rotate_quadruped_matrix = np.concatenate((rotate_quadruped_matrix, np.array([[0, 0, 0, 1]])), axis = 0)
        self.transformed_quadruped_robot_points = (transfer_quadruped_matrix @ np.linalg.inv(reset_transfer_matrix) @ rotate_quadruped_matrix @ reset_transfer_matrix @ self.quadruped_robot_points.T).T  # the transformed points of the quadruped robot
        if self.simulation_is_running and self.trajectory_steps_counter < self.K:  # if the simulation is running and the trajectory steps counter is less than the number of trajectory steps, adjust the feet positions of the quadruped robot during the simulation time
            for foot in range(self.feet_number):
                for j in range(3): self.transformed_quadruped_robot_points[foot][j] = self.quadruped_traj_feet_positions[self.trajectory_steps_counter][3 * foot + j]  # adjust the feet positions of the quadruped robot during the simulation time
    def reset_workspace(self, event = None):  # reset the workspace to its initial state
        self.scale_parameter = 1  # initialize the scale parameter of the workspace
        self.y_cor_workspace_center = 0; self.z_cor_workspace_center = 0  # initialize the coordinates of the center of the workspace
        self.rot_y_workspace = 0; self.rot_z_workspace = 0  # initialize the rotation angles of the workspace
        self.x_transfer_quadruped_com = 0; self.y_transfer_quadruped_com = 0; self.z_transfer_quadruped_com = 0  # initialize the transfer vector of the quadruped robot's center of mass
        self.rotate_quadruped_matrix = ZYX_to_R(0, 0, 0)  # initialize the rotation matrix of the quadruped robot
        self.apply_workspace_transformation()  # apply the transformation defined by the variables above to all the points of the workspace
    def transfer_workspace_start(self, event):  # initialize the coordinates of the last mouse position when the user starts to transfer the workspace
        self.last_transfer_y = event.x
        self.last_transfer_z = event.y
    def transfer_workspace(self, event):  # transfer the workspace according to the mouse movement
        self.y_cor_workspace_center = self.y_cor_workspace_center + 2*self.workspace_sensitivity * (event.x - self.last_transfer_y)
        self.z_cor_workspace_center = self.z_cor_workspace_center - 2*self.workspace_sensitivity * (event.y - self.last_transfer_z)
        self.last_transfer_y = event.x
        self.last_transfer_z = event.y
        self.apply_workspace_transformation()
    def scale_workspace(self, event):  # scale the workspace according to the mouse wheel movement
        if event.delta == -120 and self.scale_parameter >= 0.2:
            self.scale_parameter -= self.workspace_sensitivity/5
        elif event.delta == 120 and self.scale_parameter <= 15:
            self.scale_parameter += self.workspace_sensitivity/5
        self.apply_workspace_transformation()
    def rotate_workspace_start(self, event):  # initialize the coordinates of the last mouse position when the user starts to rotate the workspace
        self.last_rotation_y = event.y
        self.last_rotation_z = event.x
    def rotate_workspace(self, event):  # rotate the workspace according to the mouse movement
        self.rot_y_workspace = self.rot_y_workspace + self.workspace_sensitivity/2 * (event.y - self.last_rotation_y)
        self.rot_z_workspace = self.rot_z_workspace + self.workspace_sensitivity/2 * (event.x - self.last_rotation_z)
        self.last_rotation_y = event.y
        self.last_rotation_z = event.x
        self.apply_workspace_transformation()
    def draw_next_workspace_frame(self):  # draw the next frame of the workspace
        self.workspace.delete("all")  # clear the workspace
        # choose the points and links to draw
        self.points_to_draw = []; self.links_to_draw = []
        self.canvas_moved_points = np.array(self.canvas_moved_points); self.canvas_moved_points = self.canvas_moved_points.tolist()
        if self.axis_terrain_enable == "on":
            self.points_to_draw = self.canvas_moved_points[:self.axis_terrain_points_num] + self.quadruped_robot_points_num*[None]  # draw only the axis and terrain points
            self.links_to_draw = self.points_links[:self.axis_terrain_points_num] + self.quadruped_robot_points_num*[None]  # draw only the axis and terrain points links
        if self.quadruped_robot_enable == "on":
            if self.quadruped_points_enable == "on": self.points_to_draw = self.axis_terrain_points_num*[None] + self.canvas_moved_points[self.axis_terrain_points_num:]  # draw only the quadruped robot points
            self.links_to_draw = self.axis_terrain_points_num*[None] + self.points_links[self.axis_terrain_points_num:]  # draw only the quadruped robot points links
        if self.axis_terrain_enable == "on" and self.quadruped_robot_enable == "on":
            if self.quadruped_points_enable == "on": self.points_to_draw = self.canvas_moved_points  # draw all the points of the workspace
            else: self.points_to_draw = self.canvas_moved_points[:self.axis_terrain_points_num]  # draw only the axis and terrain points
            self.links_to_draw = self.points_links  # draw all the points links of the workspace
        # draw the terrain plane
        if self.axis_terrain_enable == "on":
            first_point = 4
            self.workspace.create_polygon([self.canvas_moved_points[first_point][0], self.canvas_moved_points[first_point][1], self.canvas_moved_points[first_point+1][0], self.canvas_moved_points[first_point+1][1], self.canvas_moved_points[first_point+2][0], self.canvas_moved_points[first_point+2][1], self.canvas_moved_points[first_point+3][0], self.canvas_moved_points[first_point+3][1]], width = 1, fill = "gray", activefill = "gray")
        # draw the chosen links (connecting lines) between a point and its neighbours
        for point in range(len(self.links_to_draw)):
            if self.links_to_draw[point] != None:
                for link in self.points_links[point]:
                    if link < self.axis_terrain_points_num:
                        link_color = "brown"
                    elif link < self.axis_terrain_points_num + 2*self.feet_number:
                        link_color = "red"
                    else:
                        link_color = "blue"
                    if point == self.axis_terrain_points_num + 3*self.feet_number and link == self.axis_terrain_points_num + 3*self.feet_number + 2:
                        link_color = "magenta"
                    self.workspace.create_line(self.canvas_moved_points[point][0], self.canvas_moved_points[point][1], self.canvas_moved_points[link][0], self.canvas_moved_points[link][1], width = 5, fill = link_color, activefill = "white")
        # draw the chosen points of the workspace (the axis, terrain points and the quadruped robot points) and create the binds for all the points of the workspace
        for point in range(len(self.points_to_draw)):
            if self.points_to_draw[point] != None:
                self.workspace.create_line(self.points_to_draw[point][0], self.points_to_draw[point][1], self.points_to_draw[point][0], self.points_to_draw[point][1], width = 10, fill = "black", capstyle = "round", activefill = "white", tags = f"point{point}")
        if self.quadruped_robot_enable == "on":
            self.workspace.create_line(self.canvas_moved_points[-1][0], self.canvas_moved_points[-1][1], self.canvas_moved_points[-1][0], self.canvas_moved_points[-1][1], width = 12, fill = "green", capstyle = "round", activefill = "white", tags = f"point{self.total_points_num-1}")
        # draw the letters x, y, z to the corresponding points of the axis
        if self.axis_terrain_enable == "on":
            self.workspace.create_text(self.canvas_moved_points[1][0]-15, self.canvas_moved_points[1][1], text = "x", font = "Calibri 15 bold", fill = "black")
            self.workspace.create_text(self.canvas_moved_points[2][0]+15, self.canvas_moved_points[2][1], text = "y", font = "Calibri 15 bold", fill = "black")
            self.workspace.create_text(self.canvas_moved_points[3][0]+15, self.canvas_moved_points[3][1], text = "z", font = "Calibri 15 bold", fill = "black")
        # loop the function
        self.workspace.after(10, self.draw_next_workspace_frame)
    def show_point_coordinates_helper(self, point):  # helper function that returns the function that shows the coordinates of the point the user's cursor is pointing to
        return lambda event: self.show_point_coordinates(point, event)
    def show_point_coordinates(self, point, event = None):  # show the coordinates of the point the user's cursor is pointing to
        self.pointing_to_point = f"({self.workspace_points[point][0]:.2f}, {self.workspace_points[point][1]:.2f}, {self.workspace_points[point][2]:.2f})"
        self.workspace.create_text(self.workspace_width/2, self.workspace_height-20, text = f"Pointing to: {self.pointing_to_point}", font = "Calibri 12 bold", fill = "black")

    def change_workspace_control_sensitivity(self, event = None):  # change the workspace mouse control sensitivity
        self.workspace_sensitivity = self.alternate_matrix_elements(self.sensitivity_values, self.workspace_sensitivity)
        self.change_control_sensitivity_button.configure(text = self.sensitivity_degrees[self.sensitivity_values.index(self.workspace_sensitivity)])
    def change_x_axis_range(self, event = None):  # change the x axis range
        self.x_axis_range = self.alternate_matrix_elements(self.axis_range_values, self.x_axis_range)
        self.change_x_axis_range_button.configure(text = self.x_axis_range)
        self.create_workspace_points_links(); self.apply_workspace_transformation()
    def change_y_axis_range(self, event = None):  # change the y axis range
        self.y_axis_range = self.alternate_matrix_elements(self.axis_range_values, self.y_axis_range)
        self.change_y_axis_range_button.configure(text = self.y_axis_range)
        self.create_workspace_points_links(); self.apply_workspace_transformation()
    def change_z_axis_range(self, event = None):  # change the z axis range
        self.z_axis_range = self.alternate_matrix_elements(self.axis_range_values, self.z_axis_range)
        self.change_z_axis_range_button.configure(text = self.z_axis_range)
        self.create_workspace_points_links(); self.apply_workspace_transformation()
    def show_axis_terrain(self, event = None):  # show or hide the axis and terrain
        self.axis_terrain_enable = self.alternate_matrix_elements(["on", "off"], self.axis_terrain_enable)
        self.show_axis_terrain_button.configure(text = self.axis_terrain_enable)
    def show_quadruped_robot(self, event = None):  # show or hide the quadruped robot
        self.quadruped_robot_enable = self.alternate_matrix_elements(["on", "off"], self.quadruped_robot_enable)
        self.show_quadruped_robot_button.configure(text = self.quadruped_robot_enable)
    def show_quadruped_points(self, event = None):  # show or hide the quadruped robot points (the edges of the model are still visible)
        self.quadruped_points_enable = self.alternate_matrix_elements(["on", "off"], self.quadruped_points_enable)
        self.show_quadruped_points_button.configure(text = self.quadruped_points_enable)
    def change_simulation_speed(self, event = None):  # change the replay simulation speed
        self.simulation_speed = self.alternate_matrix_elements(self.simulation_speed_values, self.simulation_speed)
        self.change_simulation_speed_button.configure(text = self.simulation_speed_degrees[self.simulation_speed_values.index(self.simulation_speed)])

    def change_quadruped_mass(self, event = None):  # change the quadruped robot mass
        mass = sd.askfloat("Change mass m", "Enter the quadruped robot mass (kg):", initialvalue = self.mass, minvalue = self.mass_bounds[0], maxvalue = self.mass_bounds[1], parent = self.root)
        if mass != None: self.mass = mass
    def change_quadruped_gravity(self, event = None):  # change the quadruped robot gravity
        gravity = sd.askfloat("Change gravity g", "Enter the quadruped robot gravity (m/s^2):", initialvalue = self.g, minvalue = self.gravity_bounds[0], maxvalue = self.gravity_bounds[1], parent = self.root)
        if gravity != None: self.g = gravity
    def change_quadruped_Ixx_inertia(self, event = None):  # change the quadruped robot Ixx inertia
        Ixx = sd.askfloat("Change Ixx", "Enter the quadruped robot Ixx inertia (kg*m^2):", initialvalue = self.I[0][0], minvalue = 0.0, maxvalue = self.I_components_bounds[1], parent = self.root)
        if Ixx != None: self.I[0][0] = Ixx
    def change_quadruped_Iyy_inertia(self, event = None):  # change the quadruped robot Iyy inertia
        Iyy = sd.askfloat("Change Iyy", "Enter the quadruped robot Iyy inertia (kg*m^2):", initialvalue = self.I[1][1], minvalue = 0.0, maxvalue = self.I_components_bounds[1], parent = self.root)
        if Iyy != None: self.I[1][1] = Iyy
    def change_quadruped_Izz_inertia(self, event = None):  # change the quadruped robot Izz inertia
        Izz = sd.askfloat("Change Izz", "Enter the quadruped robot Izz inertia (kg*m^2):", initialvalue = self.I[2][2], minvalue = 0.0, maxvalue = self.I_components_bounds[1], parent = self.root)
        if Izz != None: self.I[2][2] = Izz
    def change_quadruped_Ixy_inertia(self, event = None):  # change the quadruped robot Ixy inertia
        Ixy = sd.askfloat("Change Ixy", "Enter the quadruped robot Ixy (=Iyx) inertia (kg*m^2):", initialvalue = self.I[0][1], minvalue = self.I_components_bounds[0], maxvalue = self.I_components_bounds[1], parent = self.root)
        if Ixy != None: self.I[0][1] = Ixy
        self.I[1][0] = self.I[0][1]
    def change_quadruped_Ixz_inertia(self, event = None):  # change the quadruped robot Ixz inertia
        Ixz = sd.askfloat("Change Ixz", "Enter the quadruped robot Ixz (=Izx) inertia (kg*m^2):", initialvalue = self.I[0][2], minvalue = self.I_components_bounds[0], maxvalue = self.I_components_bounds[1], parent = self.root)
        if Ixz != None: self.I[0][2] = Ixz
        self.I[2][0] = self.I[0][2]
    def change_quadruped_Iyz_inertia(self, event = None):  # change the quadruped robot Iyz inertia
        Iyz = sd.askfloat("Change Iyz", "Enter the quadruped robot Iyz (=Izy) inertia (kg*m^2):", initialvalue = self.I[1][2], minvalue = self.I_components_bounds[0], maxvalue = self.I_components_bounds[1], parent = self.root)
        if Iyz != None: self.I[1][2] = Iyz
        self.I[2][1] = self.I[1][2]
    def change_body_length_x(self, event = None):  # change the body length x (body length)
        body_length_x = sd.askfloat("Change body length x", "Enter the body length x (m):", initialvalue = self.body_length_x, minvalue = self.feet_x_dist, maxvalue = np.inf, parent = self.root)
        if body_length_x != None: self.body_length_x = body_length_x
        self.calculate_draw_new_quadruped_model()
    def change_body_length_y(self, event = None):  # change the body length y (body width)
        body_length_y = sd.askfloat("Change body length y", "Enter the body length y (m):", initialvalue = self.body_length_y, minvalue = self.feet_y_dist, maxvalue = np.inf, parent = self.root)
        if body_length_y != None: self.body_length_y = body_length_y
        self.calculate_draw_new_quadruped_model()
    def change_body_length_z(self, event = None):  # change the body length z (body height)
        body_length_z = sd.askfloat("Change body length z", "Enter the body length z (m):", initialvalue = self.body_length_z, minvalue = 0.0, maxvalue = np.inf, parent = self.root)
        if body_length_z != None: self.body_length_z = body_length_z
        self.calculate_draw_new_quadruped_model()
    def adjust_quadruped_inertia(self, event = None):  # adjust the quadruped inertia I, based on the mass and body shape (rectangular paralleliped) of the robot
        if ms.askyesno("Adjust quadruped inertia", "Are you sure you want to adjust the quadruped inertia I, based on the mass and body shape (rectangular paralleliped) of the robot?"):
            self.I = np.zeros((3, 3), dtype = float)
            self.I[0][0] = self.mass * (self.body_length_y**2 + self.body_length_z**2) / 12
            self.I[1][1] = self.mass * (self.body_length_x**2 + self.body_length_z**2) / 12
            self.I[2][2] = self.mass * (self.body_length_x**2 + self.body_length_y**2) / 12
    def change_left_fore_foot_position(self, event = None):  # change the left fore foot (x, y, z) position
        left_fore_foot_x = sd.askfloat("Change left fore foot position", "Enter the left fore foot x position (m):", initialvalue = self.feet_pos[0][0], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if left_fore_foot_x != None: self.feet_pos[0][0] = left_fore_foot_x
        left_fore_foot_y = sd.askfloat("Change left fore foot position", "Enter the left fore foot y position (m):", initialvalue = self.feet_pos[0][1], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if left_fore_foot_y != None: self.feet_pos[0][1] = left_fore_foot_y
        left_fore_foot_z = sd.askfloat("Change left fore foot position", "Enter the left fore foot z position (m):", initialvalue = self.feet_pos[0][2], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if left_fore_foot_z != None: self.feet_pos[0][2] = left_fore_foot_z
        self.calculate_draw_new_quadruped_model()
    def change_feet_height(self, event = None):  # change the feet height
        feet_height = sd.askfloat("Change feet height", "Enter the feet height (m):", initialvalue = self.feet_height, minvalue = self.feet_height_bounds[0], maxvalue = self.feet_height_bounds[1], parent = self.root)
        if feet_height != None:
            self.feet_height = feet_height
        self.calculate_draw_new_quadruped_model()
    def change_dist_from_left_hind_foot(self, event = None):  # change the distance from left hind foot to left fore foot
        feet_x_dist = sd.askfloat("Change distance from left hind foot", "Enter the distance from left fore foot to left hind foot (m):", initialvalue = self.feet_x_dist, minvalue = self.feet_x_dist_bounds[0], maxvalue = self.feet_x_dist_bounds[1], parent = self.root)
        if feet_x_dist != None:
            self.feet_x_dist = feet_x_dist
            if self.feet_x_dist > self.body_length_x: self.body_length_x = self.feet_x_dist
        self.calculate_draw_new_quadruped_model()
    def change_dist_from_right_fore_foot(self, event = None):  # change the distance from right fore foot to left fore foot
        feet_y_dist = sd.askfloat("Change distance from right fore foot", "Enter the distance from left fore foot to right fore foot (m):", initialvalue = self.feet_y_dist, minvalue = self.feet_y_dist_bounds[0], maxvalue = self.feet_y_dist_bounds[1], parent = self.root)
        if feet_y_dist != None:
            self.feet_y_dist = feet_y_dist
            if self.feet_y_dist > self.body_length_y: self.body_length_y = self.feet_y_dist
        self.calculate_draw_new_quadruped_model()
    def calculate_draw_new_quadruped_model(self, event = None):  # calculate the new feet positions based on the left fore foot position, the feet height and the distances from the left fore foot to the other feet
        for foot in range(self.feet_number):  # update the feet positions
            self.feet_pos[foot][0] = self.feet_pos[0][0] - (foot%2) * self.feet_x_dist
            self.feet_pos[foot][1] = self.feet_pos[0][1] - (foot//2) * self.feet_y_dist
            self.feet_pos[foot][2] = self.feet_pos[0][2]
        self.center_of_mass = np.array([self.feet_pos[0][0] - self.feet_x_dist/2, self.feet_pos[0][1] - self.feet_y_dist/2, self.feet_pos[0][2] + self.feet_height + self.body_length_z/2])  # update the center of mass position
        self.initial_com_position[2] = self.feet_height + self.body_length_z/2; self.final_com_position[2] = self.initial_com_position[2]  # update the initial and final center of mass positions
        self.legs_bounds_x = []; self.legs_bounds_y = []; self.legs_bounds_z = []  # initialize the feet/legs bounds
        for foot in range(self.feet_number):
            self.legs_bounds_x.append([self.feet_pos[foot][0] - self.center_of_mass[0] - self.body_length_x/2, self.feet_pos[foot][0] - self.center_of_mass[0] + self.body_length_x/2])  # update the feet/legs bounds along the x-axis
            self.legs_bounds_y.append([self.feet_pos[foot][1] - self.center_of_mass[1] - self.body_length_y/2, self.feet_pos[foot][1] - self.center_of_mass[1] + self.body_length_y/2])  # update the feet/legs bounds along the y-axis
            self.legs_bounds_z.append([-1.3*(self.feet_height + self.body_length_z/2), -0.7*(self.feet_height + self.body_length_z/2)])  # update the feet/legs bounds along the z-axis
        self.create_workspace_points_links(); self.apply_workspace_transformation()
    def show_current_quadruped_robot_model(self, event = None):  # show the current quadruped robot model
        ms.showinfo("Current quadruped robot model", "The current quadruped robot model is:\n\nmass (kg) = {}\ngravity acceleration (m/s^2) = {}\ncenter of mass (com) position (m) = {} \ninertia tensor (kg*m^2) =\n{}\nleft fore (LF) foot position (m) = {}\nleft hind (LH) foot position (m) = {}\nright fore (RF) foot position (m) = {}\nright hind (RH) foot position (m) = {}\nfeet height (m) = {}\nfeet distance along the x-axis (m) = {}\nfeet distance along the y-axis (m) = {}\nbody length on the x-axis (m) = {}\nbody width on the y-axis (m) = {}\nbody height on the z-axis (m) = {}".\
                    format(self.mass, self.g, self.center_of_mass, self.I, self.feet_pos[0], self.feet_pos[1], self.feet_pos[2], self.feet_pos[3], self.feet_height, self.feet_x_dist, self.feet_y_dist, self.body_length_x, self.body_length_y, self.body_length_z))
    def get_default_quadruped_robot_model(self, event = None):  # get the default quadruped robot model
        if ms.askyesno("Get default quadruped robot model", "Are you sure you want to get the default quadruped robot model?"):
            self.mass = self.default_mass
            self.g = self.default_g
            self.I = np.copy(self.default_I)
            self.feet_pos = np.copy(self.default_feet_pos)
            self.feet_height = self.default_feet_height
            self.feet_x_dist = self.default_feet_x_dist
            self.feet_y_dist = self.default_feet_y_dist
            self.body_length_x = self.default_body_length_x
            self.body_length_y = self.default_body_length_y
            self.body_length_z = self.default_body_length_z
            self.calculate_draw_new_quadruped_model()
    
    def change_simulation_total_time(self, event = None):  # change the total time of the simulation
        self.total_time = self.alternate_matrix_elements(self.total_time_values, self.total_time)
        while int(self.total_time/self.cycles_period) * self.cycles_period != self.total_time: self.total_time = self.alternate_matrix_elements(self.total_time_values, self.total_time)
        self.change_total_time_button.configure(text = self.total_time)
        self.calculate_cycles_number(); self.calculate_gaits_number()
    def change_simulation_cycles_period(self, event = None):  # change the period of the cycles of the simulation
        self.cycles_period = self.alternate_matrix_elements(self.cycles_period_values, self.cycles_period)
        while int(self.total_time/self.cycles_period) * self.cycles_period != self.total_time or self.cycles_period < self.gaits_period: self.cycles_period = self.alternate_matrix_elements(self.cycles_period_values, self.cycles_period)
        self.change_cycles_period_button.configure(text = self.cycles_period)
        self.calculate_cycles_number(); self.calculate_gaits_number()
    def calculate_cycles_number(self, event = None):  # calculate the number of cycles
        self.cycles_number = int(self.total_time / self.cycles_period)
        self.cycles_number_indicator.configure(text = self.cycles_number)
    def change_simulation_gaits_period(self, event = None):  # change the period of the gaits of the simulation
        self.gaits_period = self.alternate_matrix_elements(self.gaits_period_values, self.gaits_period)
        while int(self.total_time/self.gaits_period) * self.gaits_period != self.total_time or int(self.cycles_period/self.gaits_period) * self.gaits_period != self.cycles_period: self.gaits_period = self.alternate_matrix_elements(self.gaits_period_values, self.gaits_period)
        self.change_gaits_period_button.configure(text = self.gaits_period)
        self.calculate_cycles_number(); self.calculate_gaits_number()
    def calculate_gaits_number(self, event = None):  # calculate the number of gaits (per foot)
        self.gaits_number = int(self.total_time / self.gaits_period)
        self.gaits_number_indicator.configure(text = self.gaits_number)
    def change_simulation_dt(self, event = None):  # change the time step of the simulation
        self.dt = self.alternate_matrix_elements(self.dt_values, self.dt)
        self.change_dt_button.configure(text = self.dt)
    def change_quadruped_initial_position(self, event = None):  # change the initial position of the quadruped robot
        initial_center_of_mass_x = sd.askfloat("Change c.o.m. initial position", "Enter the center of mass initial x position (m):", initialvalue = self.initial_com_position[0], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if initial_center_of_mass_x != None: self.initial_com_position[0] = initial_center_of_mass_x
        initial_center_of_mass_y = sd.askfloat("Change c.o.m. initial position", "Enter the center of mass initial y position (m):", initialvalue = self.initial_com_position[1], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if initial_center_of_mass_y != None: self.initial_com_position[1] = initial_center_of_mass_y
        # initial_center_of_mass_z = sd.askfloat("Change c.o.m. initial position", "Enter the center of mass initial z position (m):", initialvalue = self.initial_com_position[2], minvalue = 0.8*self.feet_height+self.body_length_z/2, maxvalue = 1.2*self.feet_height+self.body_length_z/2, parent = self.root)
        initial_center_of_mass_z = sd.askfloat("Change c.o.m. initial position", "Enter the center of mass initial z position (m):", initialvalue = self.initial_com_position[2], minvalue =  self.feet_pos_bounds[0], maxvalue =  self.feet_pos_bounds[1], parent = self.root)
        if initial_center_of_mass_z != None: self.initial_com_position[2] = initial_center_of_mass_z
    def change_quadruped_initial_orientation(self, event = None):  # change the initial orientation of the quadruped robot
        initial_body_z_rot = sd.askfloat("Change initial body z rotation", "Enter the initial body z rotation, the first rotation that is applied (degrees):", initialvalue = self.initial_body_orientation[0], minvalue = -180, maxvalue = 180, parent = self.root)
        if initial_body_z_rot != None: self.initial_body_orientation[0] = initial_body_z_rot
        initial_body_y_rot = sd.askfloat("Change initial body y rotation", "Enter the initial body y rotation, the second rotation that is applied  (degrees):", initialvalue = self.initial_body_orientation[1], minvalue = -180, maxvalue = 180, parent = self.root)
        if initial_body_y_rot != None: self.initial_body_orientation[1] = initial_body_y_rot
        initial_body_x_rot = sd.askfloat("Change initial body x rotation", "Enter the initial body x rotation, the third rotation that is applied  (degrees):", initialvalue = self.initial_body_orientation[2], minvalue = -180, maxvalue = 180, parent = self.root)
        if initial_body_x_rot != None: self.initial_body_orientation[2] = initial_body_x_rot
    def visualize_quadruped_initial_state(self, event = None):  # visualize the initial state of the quadruped robot
        self.x_transfer_quadruped_com = self.initial_com_position[0] - self.center_of_mass[0]; self.y_transfer_quadruped_com = self.initial_com_position[1] - self.center_of_mass[1]; self.z_transfer_quadruped_com = self.initial_com_position[2] - self.center_of_mass[2]
        self.rotate_quadruped_matrix = ZYX_to_R(self.initial_body_orientation[0], self.initial_body_orientation[1], self.initial_body_orientation[2])
        self.apply_workspace_transformation()
    def change_quadruped_final_position(self, event = None):  # change the final position of the quadruped robot
        final_center_of_mass_x = sd.askfloat("Change c.o.m. final position", "Enter the center of mass final x position (m):", initialvalue = self.final_com_position[0], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if final_center_of_mass_x != None: self.final_com_position[0] = final_center_of_mass_x
        final_center_of_mass_y = sd.askfloat("Change c.o.m. final position", "Enter the center of mass final y position (m):", initialvalue = self.final_com_position[1], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if final_center_of_mass_y != None: self.final_com_position[1] = final_center_of_mass_y
        # final_center_of_mass_z = sd.askfloat("Change c.o.m. final position", "Enter the center of mass final z position (m):", initialvalue = self.final_com_position[2], minvalue = 0.8*self.feet_height+self.body_length_z/2, maxvalue = 1.2*self.feet_height+self.body_length_z/2, parent = self.root)
        final_center_of_mass_z = sd.askfloat("Change c.o.m. final position", "Enter the center of mass final z position (m):", initialvalue = self.final_com_position[2], minvalue = self.feet_pos_bounds[0], maxvalue = self.feet_pos_bounds[1], parent = self.root)
        if final_center_of_mass_z != None: self.final_com_position[2] = final_center_of_mass_z
    def change_quadruped_final_orientation(self, event = None):  # change the final orientation of the quadruped robot
        final_body_z_rot = sd.askfloat("Change final body z rotation", "Enter the final body z rotation, the first rotation that is applied (degrees):", initialvalue = self.final_body_orientation[0], minvalue = -180, maxvalue = 180, parent = self.root)
        if final_body_z_rot != None: self.final_body_orientation[0] = final_body_z_rot
        final_body_y_rot = sd.askfloat("Change final body y rotation", "Enter the final body y rotation, the second rotation that is applied  (degrees):", initialvalue = self.final_body_orientation[1], minvalue = -180, maxvalue = 180, parent = self.root)
        if final_body_y_rot != None: self.final_body_orientation[1] = final_body_y_rot
        final_body_x_rot = sd.askfloat("Change final body x rotation", "Enter the final body x rotation, the third rotation that is applied  (degrees):", initialvalue = self.final_body_orientation[2], minvalue = -180, maxvalue = 180, parent = self.root)
        if final_body_x_rot != None: self.final_body_orientation[2] = final_body_x_rot
    def visualize_quadruped_final_state(self, event = None):  # visualize the final state of the quadruped robot
        self.x_transfer_quadruped_com = self.final_com_position[0] - self.center_of_mass[0]; self.y_transfer_quadruped_com = self.final_com_position[1] - self.center_of_mass[1]; self.z_transfer_quadruped_com = self.final_com_position[2] - self.center_of_mass[2]
        self.rotate_quadruped_matrix = ZYX_to_R(self.final_body_orientation[0], self.final_body_orientation[1], self.final_body_orientation[2])
        self.apply_workspace_transformation()

    def make_gaits_sequence_grid(self, event = None):  # make a new gaits sequence grid
        if event == None or (event != None and ms.askyesno("Make new gaits sequence grid", "Are you sure you want to make a new clear gaits sequence grid, based on the cycles number and gaits number simulation options?")):
            self.gaits_sequence_background.delete("all")
            self.current_total_time = self.total_time; self.current_cycles_period = self.cycles_period; self.current_gaits_period = self.gaits_period
            self.current_total_time_label.configure(text = f"Total time (sec): {self.current_total_time}"); self.current_cycles_period_label.configure(text = f"Cycles period (sec): {self.current_cycles_period}"); self.current_gaits_period_label.configure(text = f"Gaits period (sec): {self.current_gaits_period}")
            self.chosen_cycle_tens = 0; self.chosen_cycle_units = 1; self.change_chosen_cycle_tens_button.configure(text = self.chosen_cycle_tens); self.change_chosen_cycle_units_button.configure(text = self.chosen_cycle_units)
            current_gaits_number = int(self.current_total_time / self.current_gaits_period); current_cycles_number = int(self.current_total_time / self.current_cycles_period)
            gait_button_width = self.gaits_sequence_background_width / current_gaits_number; gait_button_height = self.gaits_sequence_background_height / self.feet_number
            self.gaits_sequence = []
            for i in range(self.feet_number):
                cycle_gaits = []
                for j in range(current_gaits_number):
                    cycle_gaits.append(gait_button(self.gaits_sequence_background, gait_button_width, gait_button_height, i, j, j // int(current_gaits_number / current_cycles_number)))
                    cycle_gaits[-1].set_button_on_grid(i, j)
                self.gaits_sequence.append(cycle_gaits)
    def change_chosen_cycle_tens(self, event = None):  # change the tens digit of the chosen cycle
        self.chosen_cycle_units = 0; self.chosen_cycle_tens = self.alternate_matrix_elements(list(range(0, 10)), self.chosen_cycle_tens)
        while 10 * self.chosen_cycle_tens + self.chosen_cycle_units > int(self.current_total_time / self.current_cycles_period):
            self.chosen_cycle_tens = self.alternate_matrix_elements(list(range(0, 10)), self.chosen_cycle_tens)
        if self.chosen_cycle_tens == 0: self.chosen_cycle_units = 1
        self.change_chosen_cycle_units_button.configure(text = self.chosen_cycle_units); self.change_chosen_cycle_tens_button.configure(text = self.chosen_cycle_tens)
    def change_chosen_cycle_units(self, event = None):  # change the units digit of the chosen cycle
        self.chosen_cycle_units = self.alternate_matrix_elements(list(range(0, 10)), self.chosen_cycle_units)
        while 10 * self.chosen_cycle_tens + self.chosen_cycle_units > int(self.current_total_time / self.current_cycles_period) or (self.chosen_cycle_tens == 0 and self.chosen_cycle_units == 0):
            self.chosen_cycle_units = self.alternate_matrix_elements(list(range(0, 10)), self.chosen_cycle_units)
        self.change_chosen_cycle_units_button.configure(text = self.chosen_cycle_units)
    def change_chosen_move_type(self, event = None):  # change the move type of the chosen cycle
        self.chosen_move_type = self.alternate_matrix_elements(self.move_types_list, self.chosen_move_type)
        self.change_chosen_move_type_button.configure(text = self.chosen_move_type)
    def apply_move_type_to_cycle(self, event = None):  # apply the chosen move type to the chosen cycle
        cycles_number = int(self.current_total_time / self.current_cycles_period)
        gaits_number_per_cycle = int(self.current_total_time / self.current_gaits_period / cycles_number)
        current_cycle = 10 * self.chosen_cycle_tens + self.chosen_cycle_units - 1
        current_move_type_feet_seq = self.move_types_list.index(self.chosen_move_type)
        for foot in range(self.feet_number):
            contact_gaits = []
            contact_phases = self.move_types_contact_phases[current_move_type_feet_seq][foot]
            for contact_interval in contact_phases:
                contact_gaits += list(range(int(contact_interval[0] * gaits_number_per_cycle), int(contact_interval[1] * gaits_number_per_cycle)))
            for gait in range(gaits_number_per_cycle):
                if gait in contact_gaits:
                    if self.gaits_sequence[foot][current_cycle * gaits_number_per_cycle + gait].gait_button_is_pressed == False:
                        self.gaits_sequence[foot][current_cycle * gaits_number_per_cycle + gait].press_button()
                else:
                    if self.gaits_sequence[foot][current_cycle * gaits_number_per_cycle + gait].gait_button_is_pressed == True:
                        self.gaits_sequence[foot][current_cycle * gaits_number_per_cycle + gait].press_button()
    def apply_move_type_to_all_cycles(self, event = None):  # apply the chosen move type to all the cycles
        current_cycles_number = int(self.current_total_time / self.current_cycles_period)
        for cycle in range(current_cycles_number):
            self.chosen_cycle_tens = cycle // 10; self.chosen_cycle_units = cycle % 10
            self.change_chosen_cycle_tens_button.configure(text = self.chosen_cycle_tens); self.change_chosen_cycle_units_button.configure(text = self.chosen_cycle_units)
            self.apply_move_type_to_cycle()
    def alternate_matrix_elements(self, matrix, index_element):  # alternate the parametres that are inside the matrix based on the current index_element
        return (matrix[1:] + [matrix[0]])[matrix.index(index_element)]

    def quadruped_dynamics(self, x_quad, u, contacts):  # the dynamics of the quadruped robot, based on the contacts or not (swings) of the feet with the ground
        # x_quad = [pcom, pcom_dot, q, omega, p1, p2, p3, p4]^T
        # pcom is the center of mass (body position), pcom_dot is the center of mass velocity, q is the quaternion-based representation of the body orientation, omega is the body angular velocity
        # p1, p2, p3, p4 are the positions of the feet (left fore, left hind, right fore, right hind)
        # u = [f1, f2, f3, f4]^T forces applied to the feet
        # contacts = [c1, c2, c3, c4]^T, c1 = True if the left fore foot is in contact with the ground, c1 = False otherwise, c2, c3, c4 are the same for the other feet
        # self.body_state_dim = 13, self.feet_state_dim = 12, self.body_com_dim = 6, self.body_position_dim = 3, self.feet_number = 4, self.N = 25, self.M = 12
        x_quad = x_quad.reshape((self.N, -1)); u = u.reshape((self.M, -1))  # reshape the state and the control input vectors
        pcom = x_quad[: self.body_position_dim].reshape((3, 1))  # center of mass position (body position)
        pcom_dot = x_quad[self.body_position_dim : self.body_com_dim].reshape((3, 1))  # center of mass velocity (body velocity)
        q = x_quad[self.body_com_dim : self.body_com_dim + 4].reshape((4, 1))  # quaternion-based representation of the body orientation
        omega = x_quad[self.body_com_dim + 4 : self.body_state_dim].reshape((3, 1))  # body angular velocity
        pi = x_quad[self.body_state_dim : self.N].reshape((self.feet_state_dim, 1))  # feet positions
        fi = u  # initialize the forces applied to the feet
        for foot in range(self.feet_number):  # calculate the forces applied to the feet based on the contacts
            if not contacts[foot]: fi[3 * foot : 3 * (foot + 1)] = np.zeros((3, 1))  # if the foot is in contact with the ground, the force applied to the foot is non-zero, otherwise it is zero
        F_total = sum(fi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) for foot in range(self.feet_number)) + np.array([[0., 0., -self.mass * self.g]]).reshape((3, 1))  # calculate the total force applied to the quadruped robot
        T_total = sum(hat(pi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) - pcom) @ fi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) for foot in range(self.feet_number))  # calculate the total torque applied to the quadruped robot
        pcom_ddot = F_total / self.mass  # calculate the center of mass acceleration (body acceleration)
        q_dot = 1/2 * L_matrix(q) @ np.concatenate([np.zeros((1, 1)), omega], axis = 0)  # calculate the quaternion-based representation of the body orientation derivative
        Rw = q_to_R(q)  # rotation matrix of the body orientation (equivalent to the quaternion-based representation of the body orientation)
        omega_dot = np.linalg.inv(self.I) @ (Rw.T @ T_total - hat(omega) @ (self.I @ omega))  # calculate the body angular acceleration
        return np.concatenate((pcom_dot, pcom_ddot, q_dot, omega_dot), axis = 0).reshape((self.body_state_dim, 1))  # return the body state derivative
    def quadruped_dynamics_dxquad(self, x_quad, u, contacts):  # the partial derivative of the quadruped robot dynamics with respect to the state x_quad
        x_quad = x_quad.reshape((self.N, -1)); u = u.reshape((self.M, -1))  # reshape the state and the control input vectors
        pcom = x_quad[: self.body_position_dim].reshape((3, 1))  # center of mass position (body position)
        q = x_quad[self.body_com_dim : self.body_com_dim + 4].reshape((4, 1))  # quaternion-based representation of the body orientation
        omega = x_quad[self.body_com_dim + 4 : self.body_state_dim].reshape((3, 1))  # body angular velocity
        pi = x_quad[self.body_state_dim : self.N].reshape((self.feet_state_dim, 1))  # feet positions
        fi = u  # initialize the forces applied to the feet
        for foot in range(self.feet_number):  # calculate the forces applied to the feet based on the contacts
            if not contacts[foot]: fi[3 * foot : 3 * (foot + 1)] = np.zeros((3, 1))  # if the foot is in contact with the ground, the force applied to the foot is non-zero, otherwise it is zero
        T_total = sum(hat(pi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) - pcom) @ fi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) for foot in range(self.feet_number))  # calculate the total torque applied to the quadruped robot
        Rw = q_to_R(q)  # rotation matrix of the body orientation (equivalent to the quaternion-based representation of the body orientation)
        H = np.concatenate([np.zeros((1, 3)), np.eye(3)], axis = 0)  # the H matrix used in the quaternions operations
        inv_I = np.linalg.inv(self.I)  # the inverse of the inertia tensor
        body_dyn_dxquad = np.zeros((self.body_state_dim, self.N))  # initialize the partial derivative of the quadruped robot body dynamics with respect to the state x_quad
        body_dyn_dxquad[: self.body_position_dim, self.body_position_dim : self.body_com_dim] = np.eye(self.body_position_dim)  # the partial derivative of the body velocity with respect to the body position
        body_dyn_dxquad[self.body_com_dim : self.body_com_dim + 4, self.body_com_dim: self.body_com_dim + 4] = 1/2 * np.block([[0, -omega.T], [omega, -hat(omega)]])  # the partial derivative of the quaternion derivative with respect to the quaternion
        body_dyn_dxquad[self.body_com_dim: self.body_com_dim + 4, self.body_com_dim + 4 : self.body_state_dim] = 1/2 * L_matrix(q) @ H  # the partial derivative of the quaternion derivative with respect to the body angular velocity
        body_dyn_dxquad[self.body_com_dim + 4 : self.body_state_dim, : self.body_position_dim] = inv_I @ Rw.T @ sum(hat(fi[3 * foot : 3 * (foot + 1)].reshape((3, 1))) for foot in range(self.feet_number))  # the partial derivative of the body angular acceleration with respect to the body position
        body_dyn_dxquad[self.body_com_dim + 4 : self.body_state_dim, self.body_com_dim : self.body_com_dim + 4] = inv_I @ dRTt_dq(q, T_total)  # the partial derivative of the body angular acceleration with respect to the quaternion
        body_dyn_dxquad[self.body_com_dim + 4 : self.body_state_dim, self.body_com_dim + 4 : self.body_state_dim] = inv_I @ (hat(self.I @ omega) - hat(omega) @ self.I)  # the partial derivative of the body angular acceleration with respect to the body angular velocity
        for foot in range(self.feet_number): body_dyn_dxquad[self.body_com_dim + 4 : self.body_state_dim, self.body_state_dim + 3 * foot : self.body_state_dim + 3 * (foot + 1)] = -inv_I @ Rw.T @ hat(fi[3 * foot : 3 * (foot + 1)].reshape((3, 1)))  # the partial derivative of the body angular acceleration with respect to each foot position
        return body_dyn_dxquad  # return the partial derivative of the body dynamics with respect to the state x_quad
    def quadruped_dynamics_du(self, x_quad, u, contacts):  # the partial derivative of the quadruped robot dynamics with respect to the control input u
        x_quad = x_quad.reshape((self.N, -1)); u = u.reshape((self.M, -1))  # reshape the state and the control input vectors
        pcom = x_quad[: self.body_position_dim].reshape((3, 1))  # center of mass position (body position)
        q = x_quad[self.body_com_dim : self.body_com_dim + 4].reshape((4, 1))  # quaternion-based representation of the body orientation
        pi = x_quad[self.body_state_dim : self.N].reshape((self.feet_state_dim, 1))  # feet positions
        fi = u  # initialize the forces applied to the feet
        for foot in range(self.feet_number):  # calculate the forces applied to the feet based on the contacts
            if not contacts[foot]: fi[3 * foot : 3 * (foot + 1)] = np.zeros((3, 1))  # if the foot is in contact with the ground, the force applied to the foot is non-zero, otherwise it is zero
        Rw = q_to_R(q)  # rotation matrix of the body orientation (equivalent to the quaternion-based representation of the body orientation)
        inv_I = np.linalg.inv(self.I)  # the inverse of the inertia tensor
        body_dyn_du = np.zeros((self.body_state_dim, self.M))  # initialize the partial derivative of the quadruped robot body dynamics with respect to the control input u
        for foot in range(self.feet_number): body_dyn_du[self.body_position_dim : self.body_com_dim, 3 * foot : 3 * (foot + 1)] = np.eye(3) / self.mass * contacts[foot]  # the partial derivative of the body acceleration with respect to the force applied to each foot
        for foot in range(self.feet_number): body_dyn_du[self.body_com_dim + 4 : self.body_state_dim, 3 * foot : 3 * (foot + 1)] = inv_I @ Rw.T @ hat(pi[3 * foot : 3 * (foot + 1)].reshape((3, 1)) - pcom)  # the partial derivative of the body angular acceleration with respect to the force applied to each foot
        return body_dyn_du  # return the partial derivative of the body dynamics with respect to the control input u

    def run_optimization_simulation(self, event = None):  # run the simulation and calculate the optimal trajectory for the quadruped robot
        if ms.askyesno("Run optimization/simulation", "Are you sure you want to run the optimization procedure?"):
            # calculate the initial x0 and the target x_target states
            initial_R_body = ZYX_to_R(self.initial_body_orientation[0], self.initial_body_orientation[1], self.initial_body_orientation[2])  # the initial rotation matrix of the quadruped robot's body
            initial_q_body = R_to_q(initial_R_body)  # the initial quaternion-based representation of the quadruped robot's body orientation
            self.visualize_quadruped_initial_state()  # visualize the initial state of the quadruped robot in order to set the initial state of the optimization problem
            initial_LF_foot_pos = self.workspace_points[8, :-1].reshape((3, 1))  # the initial position of the left fore foot in m
            initial_LH_foot_pos = self.workspace_points[9, :-1].reshape((3, 1))  # the initial position of the left hind foot in m
            initial_RF_foot_pos = self.workspace_points[10, :-1].reshape((3, 1))  # the initial position of the right fore foot in m
            initial_RH_foot_pos = self.workspace_points[11, :-1].reshape((3, 1))  # the initial position of the right hind foot in m
            x0 = np.block([self.initial_com_position[0], self.initial_com_position[1], self.initial_com_position[2], 0., 0., 0., initial_q_body[0], initial_q_body[1], initial_q_body[2], initial_q_body[3], 0., 0., 0., initial_LF_foot_pos.T, initial_LH_foot_pos.T, initial_RF_foot_pos.T, initial_RH_foot_pos.T]).reshape((self.N, 1))
            final_R_body = ZYX_to_R(self.final_body_orientation[0], self.final_body_orientation[1], self.final_body_orientation[2])  # the final rotation matrix of the quadruped robot's body
            final_q_body = R_to_q(final_R_body)  # the final quaternion-based representation of the quadruped robot's body orientation
            self.visualize_quadruped_final_state()  # visualize the final state of the quadruped robot in order to set the final state of the optimization problem
            final_LF_foot_pos = self.workspace_points[8, :-1].reshape((3, 1))  # the final position of the left fore foot in m
            final_LH_foot_pos = self.workspace_points[9, :-1].reshape((3, 1))  # the final position of the left hind foot in m
            final_RF_foot_pos = self.workspace_points[10, :-1].reshape((3, 1))  # the final position of the right fore foot in m
            final_RH_foot_pos = self.workspace_points[11, :-1].reshape((3, 1))  # the final position of the right hind foot in m
            x_target = np.block([self.final_com_position[0], self.final_com_position[1], self.final_com_position[2], 0., 0., 0., final_q_body[0], final_q_body[1], final_q_body[2], final_q_body[3], 0., 0., 0., final_LF_foot_pos.T, final_LH_foot_pos.T, final_RF_foot_pos.T, final_RH_foot_pos.T]).reshape((self.N, 1))

            # find the gaits sequence / feet phases for each foot and each time step of the simulation
            self.K = round(self.current_total_time / self.dt) + 1  # the total number of the knot points
            self.feet_phases = np.zeros((self.feet_number, self.K), dtype = bool)  # the gaits sequence / feet phases for each foot and each time step of the simulation
            current_gaits_number = int(self.current_total_time / self.current_gaits_period)  # the current number of gaits (per foot)
            time_steps_per_gait = int(self.current_gaits_period / self.dt)  # the number of time steps per gait
            for foot in range(self.feet_number):
                for gait in range(current_gaits_number):
                    for k in range(time_steps_per_gait):
                        self.feet_phases[foot][gait * time_steps_per_gait + k] = self.gaits_sequence[foot][gait].gait_button_is_pressed  # if the gait button is pressed, the foot is in contact with the ground, otherwise it is in swing

            # find the indexes of the contact and the swing feet phases, and the number of the feet equality and inequality constraints
            contact_indexes = [[index for index, phase in enumerate(self.feet_phases[foot]) if phase == True] for foot in range(self.feet_number)]  # the indexes of the contact feet phases
            swing_indexes = [[index for index, phase in enumerate(self.feet_phases[foot]) if phase == False] for foot in range(self.feet_number)]  # the indexes of the swing feet phases
            fix_feet_dim = 0  # initialize the number of the feet equality constraints to fix the feet positions when the feet are in contact with the ground
            feet_forces_dim = 0  # initialize the number of the feet inequality constraints to define the friction cones for the feet forces
            for foot in range(self.feet_number):
                feet_forces_dim += 4 * len(contact_indexes[foot])  # for every foot and every knot point k in contact phase, there are two friction forces (fx and fy), each of them carrying two inequality constraints
                for k in range(len(contact_indexes[foot])):
                    contact_index = contact_indexes[foot][k]
                    if contact_index < self.K - 1 and self.feet_phases[foot][contact_index + 1] == True:  # if the contact phase is not the last phase and the next phase is also a contact phase
                        fix_feet_dim += 2  # I care only about the x and y coordinates of the feet positions

            # calculate the total number of the optimization variables, the equality constraints, and the inequality constraints
            x_dim = self.K * self.N + (self.K - 1) * self.M  # the total number of the optimization variables
            eq_dim = (self.K - 1) * self.body_state_dim + fix_feet_dim + self.K  # the number of the equality constraints
            ineq_dim = feet_forces_dim + self.K * (3 * self.feet_number)  # the number of the inequality constraints

            # initialize the optimization variables (not considering the body translational and angular velocities)
            xopt0 = np.zeros((x_dim, 1))  # the initial guess for the optimization variables
            dq = (L_matrix(initial_q_body).T @ final_q_body).reshape((4, 1))  # the quaternion-based representation of the body orientation difference between the initial and the target states
            if np.abs(dq[0]) != 0.: phi_total = dq[1:] / dq[0]  # phi_total is the 3D rotation difference vector between the initial and the target states
            else: phi_total = dq[1:]
            for k in range(self.K):
                xopt0[k * self.N : k * self.N + self.body_position_dim] = x0[:self.body_position_dim] + (x_target[:self.body_position_dim] - x0[:self.body_position_dim]) * k / (self.K - 1)  # the initial guess for the body positions (center of mass) during time
                xopt0[k * self.N + self.body_state_dim : (k + 1)* self.N] = x0[self.body_state_dim : self.N] + (x_target[self.body_state_dim : self.N] - x0[self.body_state_dim : self.N]) * k / (self.K - 1)  # the initial guess for the feet positions during time
                phik = phi_total * k / (self.K - 1)  # the initial guess for the 3D rotation difference vector between the state at time k and the initial state
                dqk = np.ones((4, 1)); dqk[1:] = phik; dqk = dqk / np.linalg.norm(dqk)  # dqk is the quaternion-based representation of the body orientation difference between the state at time k and the initial state
                xopt0[k * self.N + self.body_com_dim : k * self.N + self.body_com_dim + 4] = L_matrix(initial_q_body) @ dqk  # the initial guess for the quaternion-based representation of the body orientation during time

            # define the bounds of the optimization variables
            opt_lb = x_dim * [None]  # initialize the lower bounds of the optimization variables
            opt_ub = x_dim * [None]  # initialize the upper bounds of the optimization variables
            for k in range(self.N):  # the bounds of the state optimization variables for the initial state
                opt_lb[k] = float(x0[k])
                opt_ub[k] = float(x0[k])
            for k in list(range(self.body_state_dim - 3)) + list(range(self.body_state_dim, self.N)):  # the bounds of the state optimization variables for the target state (not considering the body angular velocity)
                opt_lb[(self.K - 1) * self.N + k] = float(x_target[k])
                opt_ub[(self.K - 1) * self.N + k] = float(x_target[k])
            for foot in range(self.feet_number):  # the bounds of the state optimization variables for the z component of the feet positions
                for k in range(self.K):  # the z component of the current foot position at the current knot point must be non-negative (the foot can not penetrate the ground)
                    opt_lb[k * self.N + self.body_state_dim + 3 * foot + 2] = 0.
            for foot in range(self.feet_number):  # the bounds of the state optimization variables for the feet positions when the feet are in contact with the ground
                for k in range(len(contact_indexes[foot])):  # z component of the current foot position at the current knot point must be zero (the foot is in contact with the ground)
                    contact_index = contact_indexes[foot][k]
                    opt_lb[contact_index * self.N + self.body_state_dim + 3 * foot + 2] = 0.
                    opt_ub[contact_index * self.N + self.body_state_dim + 3 * foot + 2] = 0.
            for k in range(self.K - 1):  # the bounds of the control input optimization variables for the feet forces
                for force in range(self.M):
                    opt_lb[self.K * self.N + k * self.M + force] = float(-10 * self.mass * self.g)
                    opt_ub[self.K * self.N + k * self.M + force] = float(10 * self.mass * self.g)
            for foot in range(self.feet_number):  # the bounds of the control input optimization variables for the contact feet forces
                for k in range(len(contact_indexes[foot])):
                    contact_index = contact_indexes[foot][k]
                    opt_lb[self.K * self.N + contact_index * self.M + 3 * foot + 2] = 0.  # z component of the force applied to the current foot at the current knot point must be non-negative (the ground pushes the foot upwards)
            for foot in range(self.feet_number):  # the bounds of the control input optimization variables for the swing feet forces
                for k in range(len(swing_indexes[foot])):
                    swing_index = swing_indexes[foot][k]
                    if swing_index != self.K - 1:
                        for force in range(3 * foot, 3 * (foot + 1)):  # the forces applied to the swing feet are zero
                            opt_lb[self.K * self.N + swing_index * self.M + force] = 0.
                            opt_ub[self.K * self.N + swing_index * self.M + force] = 0.
            
            # define the bounds of the constraints
            c_lb = (eq_dim + ineq_dim) * [0.]  # initialize the lower bounds of the constraints
            c_ub = (eq_dim + ineq_dim) * [0.]  # initialize the upper bounds of the constraints
            c_index = eq_dim  # initialize the index of the inequality constraints
            for k in range(feet_forces_dim):  # the bounds of the inequality constraints for the feet forces (the friction cones)
                c_lb[c_index + k] = None
                c_ub[c_index + k] = 0.
            c_index = eq_dim + feet_forces_dim  # the index of the constraints for the feet positions
            for foot in range(self.feet_number):  # the bounds of the constraints for the feet positions with respect to the body positions (center of mass)
                for k in range(self.K):
                    c_lb[c_index + k * (3 * self.feet_number) + 3 * foot] = float(self.legs_bounds_x[foot][0])
                    c_ub[c_index + k * (3 * self.feet_number) + 3 * foot] = float(self.legs_bounds_x[foot][1])
                    c_lb[c_index + k * (3 * self.feet_number) + 3 * foot + 1] = float(self.legs_bounds_y[foot][0])
                    c_ub[c_index + k * (3 * self.feet_number) + 3 * foot + 1] = float(self.legs_bounds_y[foot][1])
                    c_lb[c_index + k * (3 * self.feet_number) + 3 * foot + 2] = float(self.legs_bounds_z[foot][0])
                    c_ub[c_index + k * (3 * self.feet_number) + 3 * foot + 2] = float(self.legs_bounds_z[foot][1])
            
            # use the cyipopt library to solve the trajectory optimization problem
            nltopt_solver = cyipopt.Problem(n = x_dim, m = eq_dim + ineq_dim, problem_obj = trajectory_optimization(self.quadruped_dynamics, self.quadruped_dynamics_dxquad, self.quadruped_dynamics_du, x0, x_target, self.K, self.dt, self.feet_phases), lb = opt_lb, ub = opt_ub, cl = c_lb, cu = c_ub)
            nltopt_solver.add_option("jacobian_approximation", "exact")  # or "finite-difference-values"
            nltopt_solver.add_option("print_level", 3)
            nltopt_solver.add_option("nlp_scaling_method", "none")
            nltopt_solver.add_option("tol", 1e-5)  # the tolerance for the convergence of the optimization algorithm
            nltopt_solver.add_option("max_iter", 100)  # the maximum number of iterations for the optimization algorithm
            xopt, info = nltopt_solver.solve(xopt0)  # solve the trajectory optimization problem and save the states that follow the optimal trajectory and obey the constraints
            self.trajectory_states_list = [xopt[k * self.N : (k + 1) * self.N] for k in range(self.K)]  # the states of the optimal trajectory
            self.inputs_list = [xopt[self.K * self.N + k * self.M : self.K * self.N + (k + 1) * self.M] for k in range(self.K - 1)]  # the control inputs of the optimal trajectory
            
            # inform the user about the optimization status
            if info['status'] == 0:
                ms.showinfo("Optimization Info", f"Successful optimization!")
            else:
                ms.showinfo("Optimization Info", f"The maximum number of iterations done. Unsuccessful optimization!")
            
            # # print some of the important states of the optimal trajectory
            # for state in self.trajectory_states_list:
            #     com_pos = state[:self.body_position_dim]
            #     q = state[self.body_com_dim : self.body_com_dim + 4]
            #     feet_pos = state[self.body_state_dim : self.N]
            #     print(f"com pos: {com_pos}")
            #     print(f"q norm: {np.linalg.norm(q)}")
            #     print(f"feet pos: {feet_pos}")
            
            # move the quadruped robot from the initial state to the final state
            self.quadruped_traj_com_locations = []
            self.quadruped_traj_body_orientations = []
            self.quadruped_traj_feet_positions = []
            for state in self.trajectory_states_list:
                self.quadruped_traj_com_locations.append(state[:self.body_position_dim])
                self.quadruped_traj_body_orientations.append(state[self.body_com_dim : self.body_com_dim + 4])
                self.quadruped_traj_feet_positions.append(state[self.body_state_dim : self.N])
            self.trajectory_steps_counter = 0
            self.show_quadruped_trajectory()

    def show_quadruped_trajectory(self, event = None):
        if self.trajectory_steps_counter < self.K:
            self.simulation_is_running = True
            self.x_transfer_quadruped_com = self.quadruped_traj_com_locations[self.trajectory_steps_counter][0] - self.center_of_mass[0]
            self.y_transfer_quadruped_com = self.quadruped_traj_com_locations[self.trajectory_steps_counter][1] - self.center_of_mass[1]
            self.z_transfer_quadruped_com = self.quadruped_traj_com_locations[self.trajectory_steps_counter][2] - self.center_of_mass[2]
            self.rotate_quadruped_matrix = q_to_R(self.quadruped_traj_body_orientations[self.trajectory_steps_counter])
            self.apply_workspace_transformation()
            self.trajectory_steps_counter += 1
            self.workspace.after(int(1000 * self.dt / self.simulation_speed), self.show_quadruped_trajectory)
        else:
            self.trajectory_steps_counter = 0
            self.simulation_is_running = False

# this class is used to find the optimal trajectory for the quadruped robot using the IPOPT solver/optimizer
class trajectory_optimization():
    def __init__(self, dynamics, dynamics_dx, dynamics_du, x0, x_target, K, dt, feet_phases):
        self.dynamics, self.dynamics_dx, self.dynamics_du = dynamics, dynamics_dx, dynamics_du  # the dynamics of the quadruped robot and their jacobians 
        self.x0 = x0  # the initial state of the quadruped robot
        self.x_target = x_target  # the target state of the quadruped robot
        self.dt = dt  # the time step of the simulation
        self.K = K  # the total number of the knot points
        self.feet_phases = feet_phases  # the gaits sequence / feet phases for each foot and each time step of the simulation
        self.feet_number = len(self.feet_phases)  # the number of the feet of the quadruped robot
        self.body_state_dim = 13  # the number of the body state variables
        self.body_com_dim = 6  # the number of the body center of mass variables (position and velocity)
        self.body_position_dim = 3  # the number of the body position variables
        self.feet_state_dim = 3 * self.feet_number  # the number of the feet state variables
        self.N = self.body_state_dim + self.feet_state_dim  # the total number of the state variables
        self.M = 3 * self.feet_number  # the total number of the control input variables
        
        # find the indexes of the contact and the swing feet phases, and the number of the feet equality and inequality constraints
        self.contact_indexes = [[index for index, phase in enumerate(self.feet_phases[foot]) if phase == True] for foot in range(self.feet_number)]  # the indexes of the contact feet phases
        self.swing_indexes = [[index for index, phase in enumerate(self.feet_phases[foot]) if phase == False] for foot in range(self.feet_number)]  # the indexes of the swing feet phases
        self.fix_feet_dim = 0  # initialize the number of the feet equality constraints to fix the feet positions when the feet are in contact with the ground
        self.feet_forces_dim = 0  # initialize the number of the feet inequality constraints to define the friction cones for the feet forces
        for foot in range(self.feet_number):
            self.feet_forces_dim += 4 * len(self.contact_indexes[foot])  # for every foot and every knot point k in contact phase, there are two friction forces (fx and fy), each of them carrying two inequality constraints
            for k in range(len(self.contact_indexes[foot])):
                contact_index = self.contact_indexes[foot][k]
                if contact_index < self.K - 1 and self.feet_phases[foot][contact_index + 1]:  # if the contact phase is not the last phase and the next phase is also a contact phase
                    self.fix_feet_dim += 2  # I care only about the x and y coordinates of the feet positions

        # define the dimensions of the optimization variables and the equality and inequality constraints
        self.x_dim = self.K * self.N + (self.K - 1) * self.M  # the size of the optimization variables
        self.eq_dim = (self.K - 1) * self.body_state_dim + self.fix_feet_dim + self.K  # the number of the equality constraints
        self.ineq_dim = self.feet_forces_dim + self.K * (3 * self.feet_number)  # the number of the inequality constraints
        
        # variables for the contacts and the friction cones
        self.mu = 1.0  # the friction coefficient
        self.tx = np.array([1., 0., 0.]).reshape((3, 1))  # the tangent vector of the contact plane of the friction cone along the x-axis
        self.ty = np.array([0., 1., 0.]).reshape((3, 1))  # the tangent vector of the contact plane of the friction cone along the y-axis
        self.nz = np.array([0., 0., 1.]).reshape((3, 1))  # the normal vector of the contact plane
        
    def objective(self, x):  # define the objective/cost function
        return 0.  # return the objective/cost function

    def gradient(self, x):  # compute the gradient of the objective/cost function
        grad = np.zeros((self.x_dim, 1))
        return grad  # return the gradient of the objective/cost function

    def constraints(self, x):  # define the constraints (equality and inequality constraints)
        x = x.reshape((self.x_dim, 1))  # reshape the optimization variables vector x to a column vector
        c = np.zeros((self.eq_dim + self.ineq_dim, 1))  # initialize the equality and inequality constraints
        
        # the dynamics equality constraints
        for k in range(self.K - 1):
            contactsk = self.feet_phases[:, k]  # the contacts of the feet at the current knot point k
            xk0 = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            xk1 = x[(k + 1) * self.N : (k + 2) * self.N]  # the state at the next knot point k + 1
            uk = x[self.K * self.N + k * self.M : self.K * self.N + (k + 1) * self.M]  # the control input at the current knot point k
            x_new = xk0[:self.body_state_dim] + self.dynamics(xk0, uk, contactsk) * self.dt  # the new state at the next knot point k + 1, using euler integration
            c[k * self.body_state_dim : (k + 1) * self.body_state_dim] = xk1[:self.body_state_dim] - x_new  # the dynamics equality constraint at the current knot point k
        
        # the feet equality constraints to fix the feet positions when the feet are in contact with the ground
        c_index = (self.K - 1) * self.body_state_dim  # the index of the feet equality constraints
        counter = 0  # the counter for the feet equality constraints
        for foot in range(self.feet_number):
            for k in range(len(self.contact_indexes[foot])):
                contact_index = self.contact_indexes[foot][k]  # the indexes for the contact feet phases of the current foot
                if contact_index < self.K - 1 and self.feet_phases[foot][contact_index + 1] == True:  # if the contact phase is not the last phase and the next phase is also a contact phase
                    footk0 = x[contact_index * self.N + self.body_state_dim + 3 * foot : contact_index * self.N + self.body_state_dim + 3 * foot + 2]  # the (x, y) terrain position of the foot at the current knot point contact_index
                    footk1 = x[(contact_index + 1) * self.N + self.body_state_dim + 3 * foot : (contact_index + 1) * self.N + self.body_state_dim + 3 * foot + 2]  # the (x, y) terrain position of the foot at the next knot point contact_index + 1
                    c[c_index + 2 * counter : c_index + 2 * (counter + 1)] = footk1 - footk0  # the feet equality constraints to fix the feet positions at the same (x, y) position on the terrain when the feet are in contact with the ground
                    counter += 1  # increase by 1 the counter for the feet equality constraints
        
        # the quaternion normalization equality constraints
        c_index = (self.K - 1) * self.body_state_dim + self.fix_feet_dim  # the index of the quaternion normalization equality constraints
        for k in range(self.K):
            xk = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            qk = xk[self.body_com_dim : self.body_com_dim + 4]  # the quaternion-based representation of the body orientation at the current knot point k
            c[c_index + k] = np.sum(qk.T @ qk) - 1.  # the quaternion normalization equality constraint at the current knot point k
        
        # the inequality constraints for the friction cones
        c_index = self.eq_dim  # the index of the inequality constraints for the friction cones
        counter = 0  # the counter for the inequality constraints for the friction cones
        for foot in range(self.feet_number):
            for k in range(len(self.contact_indexes[foot])):
                contact_index = self.contact_indexes[foot][k]  # the indexes for the contact feet phases of the current foot
                u = x[self.K * self.N + contact_index * self.M + 3 * foot : self.K * self.N + contact_index * self.M + 3 * (foot + 1)]  # the force applied to the current foot at the current knot point contact_index
                friction_force_x = np.sum(self.tx.T @ u)  # the x component of the friction force
                friction_force_y = np.sum(self.ty.T @ u)  # the y component of the friction force
                max_static_friction_force = self.mu * np.sum(self.nz.T @ u)  # the maximum static friction force
                c[c_index + 4 * counter] = friction_force_x - max_static_friction_force  # the first inequality constraint for the x component of the friction force
                c[c_index + 4 * counter + 1] = -friction_force_x - max_static_friction_force  # the second inequality constraint for the x component of the friction force
                c[c_index + 4 * counter + 2] = friction_force_y - max_static_friction_force  # the third inequality constraint for the y component of the friction force
                c[c_index + 4 * counter + 3] = -friction_force_y - max_static_friction_force  # the fourth inequality constraint for the y component of the friction force
                counter += 1  # increase by 1 the counter for the inequality constraints for the friction cones
        
        # the inequality constraints for the feet/legs bounds
        c_index = self.eq_dim + self.feet_forces_dim  # the index of the inequality constraints for the feet/legs bounds
        for k in range(self.K):
            xk = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            qk = xk[self.body_com_dim : self.body_com_dim + 4]  # the quaternion-based representation of the body orientation at the current knot point k
            Rk = q_to_R(qk)  # the rotation matrix of the body orientation at the current knot point k
            comk = xk[:self.body_position_dim]  # the center of mass position (body position) at the current knot point k
            for foot in range(self.feet_number):
                footk = xk[self.body_state_dim + 3 * foot : self.body_state_dim + 3 * (foot + 1)]  # the position of the current foot at the current knot point k
                c[c_index + k * self.feet_state_dim + 3 * foot : c_index + k * self.feet_state_dim + 3 * (foot + 1)] = Rk.T @ (footk - comk)  # the inequality constraints for the feet/legs bounds
        
        return c  # return the constraints

    def jacobian(self, x):  # compute the Jacobian of the constraints
        x = x.reshape((self.x_dim, 1))  # reshape the optimization variables vector x to a column vector
        J = np.zeros((self.eq_dim + self.ineq_dim, self.x_dim))  # initialize the Jacobian of the constraints

        # compute the Jacobian for the dynamics equality constraints
        for k in range(self.K - 1):
            contactsk = self.feet_phases[:, k]  # the contacts of the feet at the current knot point k
            xk0 = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            uk = x[self.K * self.N + k * self.M : self.K * self.N + (k + 1) * self.M]  # the control input at the current knot point k
            # the Jacobian for the dynamics equality constraints with respect to the state xk0
            J[k * self.body_state_dim : (k + 1) * self.body_state_dim, k * self.N : (k + 1) * self.N] = -np.eye(self.body_state_dim, self.N) - self.dynamics_dx(xk0, uk, contactsk) * self.dt
            # the Jacobian for the dynamics equality constraints with respect to the state xk1
            J[k * self.body_state_dim : (k + 1) * self.body_state_dim, (k + 1) * self.N : (k + 2) * self.N] = np.eye(self.body_state_dim, self.N)
            # the Jacobian for the dynamics equality constraints with respect to the control input uk
            J[k * self.body_state_dim : (k + 1) * self.body_state_dim, self.K * self.N + k * self.M : self.K * self.N + (k + 1) * self.M] = -self.dynamics_du(xk0, uk, contactsk) * self.dt

        # compute the Jacobian for the feet equality constraints to fix the feet positions when the feet are in contact with the ground
        c_index = (self.K - 1) * self.body_state_dim  # the index of the feet equality constraints
        counter = 0  # the counter for the feet equality constraints
        for foot in range(self.feet_number):
            for k in range(len(self.contact_indexes[foot])):
                contact_index = self.contact_indexes[foot][k]  # the indexes for the contact feet phases of the current foot
                if contact_index < self.K - 1 and self.feet_phases[foot][contact_index + 1] == True:  # if the contact phase is not the last phase and the next phase is also a contact phase
                    # the Jacobian for the fix feet equality constraints with respect to the feet x and y positions at the current knot point contact_index
                    J[c_index + 2 * counter : c_index + 2 * (counter + 1), contact_index * self.N + self.body_state_dim + 3 * foot : contact_index * self.N + self.body_state_dim + 3 * foot + 2] = -np.eye(2)
                    # the Jacobian for the fix feet equality constraints with respect to the feet x and y positions at the next knot point contact_index + 1
                    J[c_index + 2 * counter : c_index + 2 * (counter + 1), (contact_index + 1) * self.N + self.body_state_dim + 3 * foot : (contact_index + 1) * self.N + self.body_state_dim + 3 * foot + 2] = np.eye(2)
                    counter += 1  # increase by 1 the counter for the feet equality constraints

        # compute the Jacobian for quaternion normalization equality constraints
        c_index = (self.K - 1) * self.body_state_dim + self.fix_feet_dim  # the index of the quaternion normalization equality constraints
        for k in range(self.K):
            xk = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            qk = xk[self.body_com_dim : self.body_com_dim + 4]  # the quaternion-based representation of the body orientation at the current knot point k
            # the Jacobian for the quaternion normalization equality constraints with respect to the quaternion-based representation of the body orientation at the current knot point k
            J[c_index + k, k * self.N + self.body_com_dim : k * self.N + self.body_com_dim + 4] = 2 * qk.T

        # compute the Jacobian for the inequality constraints for the friction cones
        c_index = self.eq_dim  # the index of the inequality constraints for the friction cones
        counter = 0  # the counter for the inequality constraints for the friction cones
        for foot in range(self.feet_number):
            for k in range(len(self.contact_indexes[foot])):
                contact_index = self.contact_indexes[foot][k]  # the indexes for the contact feet phases of the current foot
                # the Jacobian for the inequality constraints for the friction cones with respect to the force applied to the current foot at the current knot point contact_index
                J[c_index + 4 * counter, self.K * self.N + contact_index * self.M + 3 * foot : self.K * self.N + contact_index * self.M + 3 * (foot + 1)] = (self.tx - self.mu * self.nz).T
                J[c_index + 4 * counter + 1, self.K * self.N + contact_index * self.M + 3 * foot : self.K * self.N + contact_index * self.M + 3 * (foot + 1)] = (-self.tx - self.mu * self.nz).T
                J[c_index + 4 * counter + 2, self.K * self.N + contact_index * self.M + 3 * foot : self.K * self.N + contact_index * self.M + 3 * (foot + 1)] = (self.ty - self.mu * self.nz).T
                J[c_index + 4 * counter + 3, self.K * self.N + contact_index * self.M + 3 * foot : self.K * self.N + contact_index * self.M + 3 * (foot + 1)] = (-self.ty - self.mu * self.nz).T
                counter += 1  # increase by 1 the counter for the inequality constraints for the friction cones

        # compute the Jacobian for the inequality constraints for the feet/legs bounds
        c_index = self.eq_dim + self.feet_forces_dim  # the index of the inequality constraints for the feet/legs bounds
        for k in range(self.K):
            xk = x[k * self.N : (k + 1) * self.N]  # the state at the current knot point k
            qk = xk[self.body_com_dim : self.body_com_dim + 4]  # the quaternion-based representation of the body orientation at the current knot point k
            Rk = q_to_R(qk)  # the rotation matrix of the body orientation at the current knot point k
            comk = xk[:self.body_position_dim]  # the center of mass position (body position) at the current knot point k
            for foot in range(self.feet_number):
                footk = xk[self.body_state_dim + 3 * foot : self.body_state_dim + 3 * (foot + 1)]  # the position of the current foot at the current knot point k
                # the Jacobian for the inequality constraints for the feet/legs bounds with respect to the body position at the current knot point k
                J[c_index + k * self.feet_state_dim + 3 * foot : c_index + k * self.feet_state_dim + 3 * (foot + 1), k * self.N : k * self.N + self.body_position_dim] = -Rk.T
                # the Jacobian for the inequality constraints for the feet/legs bounds with respect to the feet position at the current knot point k
                J[c_index + k * self.feet_state_dim + 3 * foot : c_index + k * self.feet_state_dim + 3 * (foot + 1), k * self.N + self.body_state_dim + 3 * foot : k * self.N + self.body_state_dim + 3 * (foot + 1)] = Rk.T
                # the Jacobian for the inequality constraints for the feet/legs bounds with respect to the quaternion-based representation of the body orientation at the current knot point k
                J[c_index + k * self.feet_state_dim + 3 * foot : c_index + k * self.feet_state_dim + 3 * (foot + 1), k * self.N + self.body_com_dim : k * self.N + self.body_com_dim + 4] = dRTt_dq(qk, footk - comk)
        
        return J  # return the Jacobian of the constraints

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):  # print info
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))  # print the objective value for each iteration


# this class creates instances of the gait (foot phase) buttons
class gait_button():
    press_colors = ["red", "yellow", "brown", "magenta"]
    unpress_colors = ["#000077", "#0000ff"]
    highlight_color = "white"
    enter_button_state = "highlight"
    continuous_paint_state = "mark"
    background_offset = 3
    def __init__(self, grid_background, gait_button_width, gait_button_height, gait_foot_index, gait_time_index, gait_cycle_index):
        self.grid_background = grid_background  # the frame that contains the sequence grid
        self.gait_button_width = gait_button_width  # the width of the gait button
        self.gait_button_height = gait_button_height  # the height of the gait button
        self.gait_foot_index = gait_foot_index  # the foot (the sequence grid row too) to which the gait belongs
        self.gait_time_index = gait_time_index  # the time moment (the sequence grid column too) that the gait refers to
        self.gait_cycle_index = gait_cycle_index  # the moving cycle to which the gait belongs
        self.control_highlight = 0  # 0 for highlighting the gait button, 1 for not highlighting
        self.gait_button_is_pressed = False  # indicator for the gait button pressing state, it also works as an indicator for the foot phase (False/0 for swing and True/1 for contact)
    def set_button_on_grid(self, row, column):  # set the gait button on the gaits sequence grid
        self.button = self.grid_background.create_rectangle([column * self.gait_button_width + gait_button.background_offset, row * self.gait_button_height + gait_button.background_offset, \
                                                            (column + 1) * self.gait_button_width + gait_button.background_offset, (row + 1) * self.gait_button_height + gait_button.background_offset], width = 1, \
                                                            fill = gait_button.unpress_colors[self.gait_cycle_index % 2], outline = "black", tags = f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}")
        self.grid_background.tag_bind(f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}", "<Button-1>", self.press_button)
        self.grid_background.tag_bind(f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}", "<Button-2>", self.change_continuous_paint_state)
        self.grid_background.tag_bind(f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}", "<Button-3>", self.change_enter_button_mode)
        self.grid_background.tag_bind(f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}", "<Enter>", self.highlight_button_paint_continuously)
        self.grid_background.tag_bind(f"button{self.gait_cycle_index}_{self.gait_foot_index}_{self.gait_time_index}", "<Leave>", self.unhighlight_button)
    def change_enter_button_mode(self, event = None):  # change the gait button color mode (highlight or paint) when the right mouse button is pressed
        if gait_button.enter_button_state == "highlight":
            gait_button.enter_button_state = "paint"
        elif gait_button.enter_button_state == "paint":
            gait_button.enter_button_state = "highlight"
    def change_continuous_paint_state(self, event = None):  # change the continuous paint state (mark or erase meaning pressing or unpressing the gait button) when the middle mouse button is pressed
        if gait_button.continuous_paint_state == "mark":
            gait_button.continuous_paint_state = "erase"
        elif gait_button.continuous_paint_state == "erase":
            gait_button.continuous_paint_state = "mark"
    def highlight_button_paint_continuously(self, event = None):  # when the mouse pointer enters the gait button area, highlight the gait button or paint it continuously based on the current enter button state and continuous paint state
        if gait_button.enter_button_state == "highlight":
            self.grid_background.itemconfigure(self.button, fill = gait_button.highlight_color)
        if gait_button.enter_button_state == "paint":
            if gait_button.continuous_paint_state == "mark":
                if not self.gait_button_is_pressed:
                    self.press_button()
            elif gait_button.continuous_paint_state == "erase":
                if self.gait_button_is_pressed:
                    self.press_button()
    def unhighlight_button(self, event = None):  # when the mouse pointer leaves the gait button area, unhighlight the gait button
        try:
            if self.control_highlight == 0:
                self.grid_background.itemconfigure(self.button, fill = gait_button.unpress_colors[self.gait_cycle_index % 2])
            elif self.control_highlight == 1:
                self.grid_background.itemconfigure(self.button, fill = gait_button.press_colors[self.gait_foot_index])
        except:
            pass
    def press_button(self, event = None):  # press the gait button (meaning contact) or unpress it (meaning swing) based on the current gait button pressing state
        if self.gait_button_is_pressed:
            self.grid_background.itemconfigure(self.button, fill = gait_button.unpress_colors[self.gait_cycle_index % 2])
            self.control_highlight = 0
        else:
            self.grid_background.itemconfigure(self.button, fill = gait_button.press_colors[self.gait_foot_index])
            self.control_highlight = 1
        self.gait_button_is_pressed = not self.gait_button_is_pressed


# this class creates instances of menu button units
class menu_button():
    def __init__(self, background, button_text, button_font, button_fg, button_bg, button_xcor, button_ycor, button_func):
        self.button = tk.Label(master = background, text = button_text, font = button_font, fg = button_fg, bg = button_bg)
        self.button.place(x = button_xcor, y = button_ycor, anchor = "center")
        self.button.bind("<Enter>", lambda event, button = self.button: button.configure(font = "{} {} bold".format(button["font"].split(" ")[0], int(button["font"].split(" ")[1]) + 10)))
        self.button.bind("<Leave>", lambda event, button = self.button: button.configure(font=button_font))
        self.button.bind("<Button-1>", lambda event: button_func(event))


# this class creates instances of menu labels
class menu_label():
    def __init__(self, background, label_text, label_font, label_fg, label_bg, label_xcor, label_ycor):
        self.label = tk.Label(master = background, text = label_text, font = label_font, fg = label_fg, bg = label_bg)
        self.label.place(x=label_xcor, y=label_ycor, anchor = "center")


# the global functions below are needed for the quadruped robot simulation
def hat(vector):  # skew-symmetric matrix of the vector vec
    v = vector.reshape((3,))
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])  # return the skew-symmetric matrix of the vector
def L_matrix(q):  # the L(q) function
    q = q.reshape((4, -1)); s = q[0]; v = q[1:]  # extract the elements of the quaternion q
    L = np.block([[s, -v.T], [v, s * np.eye(3) + hat(v)]])
    return L  # return the 4x4 matrix L(q)
def R_matrix(q):  # the R(q) function
    q = q.reshape((4, -1)); s = q[0]; v = q[1:]  # extract the elements of the quaternion q
    R = np.block([[s, -v.T], [v, s * np.eye(3) - hat(v)]])
    return R  # return the 4x4 matrix R(q)
def q_to_R(q):  # convert the quaternion q to the corresponding rotation matrix R
    q = q.reshape((4,))
    s = q[0]; v1 = q[1]; v2 = q[2]; v3 = q[3]  # extract the elements of the quaternion q
    r00 = 2. * (s**2 + v1**2) - 1.; r01 = 2. * (v1 * v2 - s * v3); r02 = 2. * (v1 * v3 + s * v2)  # the first row of the rotation matrix
    r10 = 2. * (v1 * v2 + s * v3); r11 = 2. * (s**2 + v2**2) - 1.; r12 = 2. * (v2 * v3 - s * v1)  # the second row of the rotation matrix
    r20 = 2. * (v1 * v3 - s * v2); r21 = 2. * (v2 * v3 + s * v1); r22 = 2. * (s**2 + v3**2) - 1.  # the third row of the rotation matrix
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])  # return the 3x3 rotation matrix
def q_to_R_2(q):  # convert the quaternion q to the corresponding rotation matrix R
    H = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return H.T @ L_matrix(q) @ R_matrix(q).T @ H  # return the 3x3 rotation matrix
def R_to_q(R):  # convert the rotation matrix R to the corresponding quaternion q
    r00 = R[0, 0]; r01 = R[0, 1]; r02 = R[0, 2]; r10 = R[1, 0]; r11 = R[1, 1]; r12 = R[1, 2]; r20 = R[2, 0]; r21 = R[2, 1]; r22 = R[2, 2]  # extract the elements of the rotation matrix R
    s = 1/2 * np.sqrt(1 + r00 + r11 + r22)  # the scalar part of the quaternion q
    v1 = (r21 - r12) / (4 * s)  # the first element of the vector part of the quaternion q
    v2 = (r02 - r20) / (4 * s)  # the second element of the vector part of the quaternion q
    v3 = (r10 - r01) / (4 * s)  # the third element of the vector part of the quaternion q
    return np.array([s, v1, v2, v3])  # return the quaternion q
def ZYX_to_R(z, y, x):  # convert the ZYX Euler angles to the corresponding rotation matrix R
    x = np.deg2rad(x); y = np.deg2rad(y); z = np.deg2rad(z)  # convert the Euler angles to radians
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])  # the rotation matrix around the x-axis
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])  # the rotation matrix around the y-axis
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])  # the rotation matrix around the z-axis
    return Rx @ Ry @ Rz  # return the 3x3 rotation matrix
def dRTt_dq(q, t):  # the partial derivative of the vector R^T(q)*t (where R^T(q) is the transpose of the rotation matrix R(q)) with respect to the quaternion q
    q = q.reshape((4,)); t = t.reshape((3,))
    s = q[0]; v1 = q[1]; v2 = q[2]; v3 = q[3]; t1 = t[0]; t2 = t[1]; t3 = t[2]  # extract the elements of the quaternion q and the vector t
    deriv = np.zeros((3, 4))
    deriv[0, 0] = 4. * s * t1 + 2. * v3 * t2 - 2. * v2 * t3; deriv[0, 1] = 4. * v1 * t1 + 2. * v2 * t2 + 2. * v3 * t3; deriv[0, 2] = 2. * v1 * t2 - 2. * s * t3; deriv[0, 3] = 2. * s * t2 + 2. * v1 * t3  # the first row
    deriv[1, 0] = -2. * v3 * t1 + 4. * s * t2 + 2. * v1 * t3; deriv[1, 1] = 2. * v2 * t1 + 2. * s * t3; deriv[1, 2] = 2. * v1 * t1 + 4. * v2 * t2 + 2. * v3 * t3; deriv[1, 3] = -2. * s * t1 + 2. * v2 * t3  # the second row
    deriv[2, 0] = 2. * v2 * t1 - 2. * v1 * t2 + 4. * s * t3; deriv[2, 1] = 2. * v3 * t1 - 2. * s * t2; deriv[2, 2] = 2. * s * t1 + 2. * v3 * t2; deriv[2, 3] = 2. * v1 * t1 + 2. * v2 * t2 + 4. * v3 * t3  # the third row
    return deriv  # return the partial derivative of the vector R^T(q)*t with respect to the quaternion q

# the code block below is used to create the quadruped robot simulation API window or windows, depending on the number of the program instances (windows) the user wants to be created
if __name__ == "__main__":
    windows_number = int(input("How many windows (program instances) do you want to be created? "))
    # windows_number = 1
    roots_list = []
    apis_list = []
    for window in range(windows_number):
        roots_list.append(tk.Tk())
        apis_list.append(quadruped_robot_api(roots_list[window], window))
    for window in range(windows_number):
        roots_list[window].mainloop()