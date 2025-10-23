import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import atan2, sqrt, cos, sin, pi

np.random.seed(42) 
plt.ion()
fig, ax = plt.subplots()

# robot_pose = np.array([1.156, 3.958, np.deg2rad(58.0)])
landmarks_true = np.array([
    [3.374, 2.247],
    [7.951, 4.788],
    [5.599, 2.134],
    [1.559, 0.407],
    [8.729, 5.852]
]) 
ball_pose = np.array([4.742, 2.862])

fov_rad = np.deg2rad(180.0) 
half_fov = fov_rad / 2.0

def wrap_angle(a):
    while a > pi: a -= 2*pi
    while a <= -pi: a += 2*pi
    return a

def rmse(a, b):
    dif = a-b
    return np.linalg.norm(dif)

def ekfUpdate(robot_pose, measurements, P_init, Q_lm, R):
    theta = robot_pose[2]
    x_init = robot_pose[0] + measurements[i][0] * cos(theta + measurements[i][1])
    y_init = robot_pose[1] + measurements[i][0] * sin(theta + measurements[i][1])
    init_est = np.array([x_init, y_init])
    dx = init_est[0] - robot_pose[0]
    dy = init_est[1] - robot_pose[1]
    q = dx*dx + dy*dy
    if q < 1e-8: q = 1e-8
    pred_r = sqrt(q)
    pred_b = atan2(dy, dx) - robot_pose[2]
    pred_b = wrap_angle(pred_b)
    P = P_init.copy()
    P_pred = P + Q_lm
    x_pred = init_est.copy()
    H = np.array([[dx/pred_r, dy/pred_r],
                [-dy/q, dx/q]])
    zvec = np.array([measurements[i][0], measurements[i][1]])
    hvec = np.array([pred_r, pred_b])
    y = zvec - hvec
    y[1] = wrap_angle(y[1])
    S = H @ P @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    I = np.identity(2)
    P_upd = (I - K @ H) @ P_pred
    return (x_upd, P_upd)

def computeTrueMeasurement(robot_pose, landmark_pose):
    dx = landmark_pose[0] - robot_pose[0]
    dy = landmark_pose[1] - robot_pose[1]
    r = sqrt(dx**2 + dy**2)
    b = atan2(dy, dx) - robot_pose[2]
    b = wrap_angle(b)
    return (r, b)

def ekfUpdateSingle(robot_pose, noisyData, P_init, Q_lm, R):
    theta = robot_pose[2]
    x_init = robot_pose[0] + noisyData[0] * cos(theta + noisyData[1])
    y_init = robot_pose[1] + noisyData[0] * sin(theta + noisyData[1])
    init_est = np.array([x_init, y_init])
    dx = init_est[0] - robot_pose[0]
    dy = init_est[1] - robot_pose[1]
    q = dx*dx + dy*dy
    if q < 1e-8: q = 1e-8
    pred_r = sqrt(q)
    pred_b = atan2(dy, dx) - robot_pose[2]
    pred_b = wrap_angle(pred_b)
    P = P_init.copy()
    P_pred = P + Q_lm
    x_pred = init_est.copy()
    H = np.array([[dx/pred_r, dy/pred_r],
                [-dy/q, dx/q]])
    zvec = np.array([noisyData[0], noisyData[1]])
    hvec = np.array([pred_r, pred_b])
    y = zvec - hvec
    y[1] = wrap_angle(y[1])
    S = H @ P @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    I = np.identity(2)
    P_upd = (I - K @ H) @ P_pred
    return (x_upd, P_upd)


#Tuning measurement noise
sigma_r = 0.20
sigma_b_deg = 0.05
sigma_b = np.deg2rad(sigma_b_deg)

R = np.diag([sigma_r**2, sigma_b**2])

num_runs = 5  #adjustable run
all_runs = []

for step in range(20):
    robot_coordx = np.random.uniform(0.50, 8.55)
    robot_coordy = np.random.uniform(0.50, 5.75)
    robot_angle = np.random.uniform(-3.10, 3.10)
    robot_pose = np.array([robot_coordx, robot_coordy, robot_angle])
    ax.clear()
    ax.set_title(f"Step {step+1}")
    ax.set_xlim(0,9)
    ax.set_ylim(0,6)
    dx = ball_pose[0] - robot_pose[0]
    dy = ball_pose[1] - robot_pose[1]
    robot_pose[2] = atan2(dy, dx)
    #ngitung FOV
    fov_length = 10 
    theta = robot_pose[2]
    left_angle = theta + half_fov
    right_angle = theta - half_fov
    left_x = robot_pose[0] + fov_length * np.cos(left_angle)
    left_y = robot_pose[1] + fov_length * np.sin(left_angle)
    right_x = robot_pose[0] + fov_length * np.cos(right_angle)
    right_y = robot_pose[1] + fov_length * np.sin(right_angle)

    #show FOV part
    arrow_length = 1.0
    ax.arrow(
        robot_pose[0],
        robot_pose[1],
        arrow_length * np.cos(robot_pose[2]),
        arrow_length * np.sin(robot_pose[2]),
        head_width=0.2,
        head_length=0.3,
        fc='red',
        ec='black'
    )

    ball_vel = np.random.uniform(-0.28, 0.28, size = 2)
    ball_dot = ax.scatter([], [], c='purple', s=100, marker='o', label='Ball')
    (line_to_ball,) = ax.plot([], [], 'k--', alpha=0.4, label='Ball Connection')
    decay = 0.85
    #Ball moving + estimating with EKF (later)
    for sub in range(10):
        ball_pose+=ball_vel
        ball_vel*=decay
        if ball_pose[0] >= 8.8 or ball_pose[0] <= 0.2:
            ball_vel[0] *= -1  
        if ball_pose[1] >= 5.8 or ball_pose[1] <= 0.2:
            ball_vel[1] *= -1  

        range2Ball, bear2Ball = computeTrueMeasurement(robot_pose, ball_pose)
        ballNoisy_r = range2Ball + np.random.normal(0, sigma_r)
        ballNoisy_b = bear2Ball + np.random.normal(0, sigma_b)
        P_init = np.eye(2)*8.4
        Q_lm = np.eye(2)*1e-3
        estBall, covBall = ekfUpdateSingle(robot_pose, [ballNoisy_r, ballNoisy_b], P_init, Q_lm, R)

        ax.scatter(estBall[0], estBall[1], c='green', marker='x', s=100)
        ax.plot(robot_pose[0], robot_pose[1], 'r^', markersize=12, label='Robot' if sub == 0 else "")
        ax.scatter(landmarks_true[:,0], landmarks_true[:,1], c='blue', s=70, label='Landmarks' if sub == 0 else "")
        ball_dot.set_offsets(ball_pose)
        ax.plot([robot_pose[0], left_x], [robot_pose[1], left_y], 'orange', linestyle='--', alpha=0.5)
        ax.plot([robot_pose[0], right_x], [robot_pose[1], right_y], 'orange', linestyle='--', alpha=0.5)
        ax.plot([], [], 'orange', linestyle='--', alpha=0.5, label='Field of View' if sub == 0 else "")
        line_to_ball.set_data([robot_pose[0], ball_pose[0]], [robot_pose[1], ball_pose[1]])
        ax.scatter([], [], c='green', marker='x', s=100, label="EKF Estimates" if sub == 0 else "")
        ax.legend(loc='upper right')
        plt.pause(0.2)

    #Estimating landmark with EKF
    for run in range(num_runs):
        final_est = []
        measurements = []
        visible = []
        for i, (lx, ly) in enumerate(landmarks_true):
            r, b = computeTrueMeasurement(robot_pose, [lx, ly])

            if abs(b) - half_fov < 1e-8:
                visible.append(i)
                noisy_r = r + np.random.normal(0, sigma_r)
                noisy_b = b + np.random.normal(0, sigma_b)
                wrap_angle(noisy_b)
                measurements.append((noisy_r, noisy_b))
            else:
                measurements.append((None, None))

        P_init = np.eye(2)*3.2
        Q_lm = np.eye(2)*1e-5
        for i, (lx, ly) in enumerate(landmarks_true):
            if measurements[i][0] is None:
                # final_est.append({
                #     "landmark_id": i+1,
                #     # "true_x": lx,
                #     # "true_y": ly,
                #     "final_cov_x": 100.0,
                #     "final_cov_y": 100.0,
                #     "rmse": np.nan,
                #     "final_est_x": np.nan,
                #     "final_est_y": np.nan
                # })
                continue
            ax.plot([robot_pose[0], lx], [robot_pose[1], ly], 'k--', alpha=0.4) #line from robot to landmark
            est, cov = ekfUpdate(robot_pose, measurements, P_init, Q_lm, R)
            ax.scatter(est[0], est[1], c='green', marker='x', s=100)
            plt.pause(0.1)
            # final_est.append({
            #     "landmark_id": i+1,
            #     # "true_x": lx,
            #     # "true_y": ly,
            #     "final_cov_x": covarVal[0][0],
            #     "final_cov_y": covarVal[1,1],
            #     "rmse": rmse(estimatedCoord, np.array([lx,ly])),
            #     "final_est_x": estimatedCoord[0],
            #     "final_est_y": estimatedCoord[1]
            # })
    plt.pause(0.5)

    final_df = pd.DataFrame(final_est)
    all_runs.append(final_df)

# Combine all runs
results_df = pd.concat(all_runs, ignore_index=True)

mean_rmse = results_df.groupby("landmark_id")["rmse"].mean()
print(mean_rmse)

# print(results_df)
