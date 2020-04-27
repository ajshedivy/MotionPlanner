
import scipy.optimize
import scipy.integrate
from math import sin, cos, pi, sqrt
import sys
sys.path.insert(
    0, r"C:\\Users\\ajshe\\OneDrive\\Documents\\autonomous\\MotionPlanner\\MotionPlannerCourse4FinalProject")
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser 
import local_planner
import behavioural_planner

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 100.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'course4_waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
NUM_PATHS = 7
BP_LOOKAHEAD_BASE      = 16.0              # m
BP_LOOKAHEAD_TIME      = 1.0              # s
PATH_OFFSET            = 1.5              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the 
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.
obsticles = [0,0,0,0]

# Course 4 specific parameters
C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5        # m
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'




def main():

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_filepath =\
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        WAYPOINTS_FILENAME)
    waypoints_np   = None
    with open(waypoints_filepath) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle, 
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints_np = np.array(waypoints)

    #############################################
    # Local Planner Variables
    #############################################
    wp_goal_index   = 0
    local_waypoints = None
    path_validity   = np.zeros((NUM_PATHS, 1), dtype=bool)
    lp = local_planner.LocalPlanner(NUM_PATHS,
                                    PATH_OFFSET,
                                    CIRCLE_OFFSETS,
                                    CIRCLE_RADII,
                                    PATH_SELECT_WEIGHT,
                                    TIME_GAP,
                                    A_MAX,
                                    SLOW_SPEED,
                                    STOP_LINE_BUFFER)
    bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                obsticles,
                                                LEAD_VEHICLE_LOOKAHEAD)
    ego_state = [0, 0, 0, 0, 0]
    current_speed = 10

    # Perform a state transition in the behavioural planner.
    bp.transition_state(waypoints, ego_state, current_speed)

    # Compute the goal state set from the behavioural planner's computed goal state.
    goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)
    print("goal State:", goal_state_set)

    # Calculate planned paths in the local frame.
    paths, path_validity = lp.plan_paths(goal_state_set)
    print("paths: ", paths)
    

    # Transform those paths back to the global frame.
    paths = local_planner.transform_paths(paths, ego_state)
    print("transformed paths:", paths)

    # Local path plotter update
    
    path_counter = 0
    for i in range(NUM_PATHS):
        # If a path was invalid in the set, there is no path to plot.
        if path_validity[i]:
            # Colour paths according to collision checking.
            colour = 'b'
            # trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
            plt.plot(paths[path_counter][0], paths[path_counter][1], colour)
            path_counter += 1
        else:
            continue
            # trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
    plt.show()


if __name__ == "__main__":
    main()
