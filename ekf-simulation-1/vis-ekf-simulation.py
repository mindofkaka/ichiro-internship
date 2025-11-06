import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, TextBox
from math import sin, cos, atan2, sqrt

np.random.seed()

# Parameter
dt = 0.05
n_landmarks = 3
landmarks = np.random.uniform(low=[3.0, -3.0], high=[9.0, 3.0], size=(n_landmarks, 2))
colors = plt.cm.tab10(np.arange(n_landmarks))

robot_gt = np.array([0.0, 0.0, 0.0])
robot_odom = robot_gt.copy()
robot_path = []

ball_pos = np.array([6.0, 0.0])
ball_vel = np.random.randn(2) * 2

v, omega = 1.5, 0.1

odom_pos_noise = 0.02
odom_yaw_noise = 0.01
meas_range_noise = 0.375
meas_bearing_noise = np.deg2rad(4.2)

x_min, x_max = 0, 10
y_min, y_max = -5, 5

def wrap_angle(angle):
    while angle > np.pi:
        angle -= 2.0*np.pi
    while angle < -np.pi:
        angle += 2.0*np.pi
    return angle

class EKFStatic2D:
    def __init__(self, x0, P0):
        self.x = x0.copy().reshape(2)
        self.P = P0.copy()
        self.Q = np.eye(2) * 1e-12
        self.R = np.diag([0.0, 0.0])

    def predict(self, dt):
        self.P = self.P + self.Q * dt

    def update(self, z_global, robot_pose):
        xr, yr, theta = robot_pose
        x_lm, y_lm = self.x[0], self.x[1]

        # 1. Hitung Model Pengukuran Prediksi h(x)
        dx = x_lm - xr
        dy = y_lm - yr 
        q = dx**2 + dy**2

        if q < 0.04: return

        if q < 1e-8: q = 1e-8
        
        pred_r = sqrt(q) 
        pred_b = atan2(dy, dx) - theta
        pred_b = wrap_angle(pred_b)
        
        h_vec = np.array([pred_r, pred_b]) # h(x)

        # 2. Hitung Jacobian H
        H = np.array([
            [dx/pred_r, dy/pred_r],  # Turunan range
            [-dy/q, dx/q]           # Turunan bearing
        ])

        # 3. Hitung Inovasi y
        z_vec = z_global.copy()

        y = z_vec - h_vec

        base_sigma_r = meas_range_noise
        base_sigma_b = meas_bearing_noise
        distance = sqrt(q)
        adaptive_sigma_r = base_sigma_r * (1 + 0.05 * distance)  # increase with distance
        adaptive_sigma_b = base_sigma_b * (1 + 0.3 * abs(pred_b)) # increase with angle
        self.R = np.diag([adaptive_sigma_r**2, adaptive_sigma_b**2])
        # 4. Update (S, K, x, P) - sama seperti EKF standar
        S = H @ self.P @ H.T + self.R
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        threshold = 5.99 # 99% for 2 DOF
        if d2 > threshold: return
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P
        

class EKFBall:
    def __init__(self, x0, P0):
        self.x = x0.copy().reshape(4)
        self.P = P0.copy()
        self.Q = np.diag([0.002, 0.002, 0.05, 0.05])
        self.R = np.diag([0.0,0.0])

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, robot_pose):
        xr, yr, theta = robot_pose 
        x_lm, y_lm = self.x[0], self.x[1]

        dx = x_lm - xr
        dy = y_lm - yr
        q = dx**2 + dy**2

        if q < 0.01: return

        if q < 1e-8: q = 1e-8
        pred_r = sqrt(q)
        pred_b = atan2(dy, dx) - theta
        pred_b = wrap_angle(pred_b)

        h_vec = np.array([pred_r, pred_b])

        H = np.array([
            [dx/pred_r, dy/pred_r, 0, 0],
            [-dy/q, dx/q, 0, 0]
        ])
        z_vec = z.copy()
        y = z_vec - h_vec

        base_sigma_r = meas_range_noise
        base_sigma_b = meas_bearing_noise
        distance = sqrt(q)
        adaptive_sigma_r = base_sigma_r * (1 + 0.05 * distance)  # increase with distance
        adaptive_sigma_b = base_sigma_b * (1 + 0.3 * abs(pred_b)) # increase with angle
        self.R = np.diag([adaptive_sigma_r**2, adaptive_sigma_b**2])
        S = H @ self.P @ H.T + self.R
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        threshold = 5.99 # 99% for 2 DOF
        if d2 > threshold: return
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


ekfs = []
for lm in landmarks:
    P0 = np.eye(2) * 5.0
    ekfs.append(EKFStatic2D(lm + np.random.randn(2)*2.0, P0))

ekf_ball = EKFBall(np.array([6.0, 0.0, 0.0, 0.0]), np.eye(4)*5.0)

def robot_to_global(pose, pt_robot):
    x, y, th = pose
    c, s = cos(th), sin(th)
    gx = x + c * pt_robot[0] - s * pt_robot[1]
    gy = y + s * pt_robot[0] + c * pt_robot[1]
    return np.array([gx, gy])

def global_to_robot(pose, pt_global):
    x, y, th = pose
    dx, dy = pt_global[0] - x, pt_global[1] - y
    c, s = cos(th), sin(th)
    rx = c*dx + s*dy
    ry = -s*dx + c*dy
    return np.array([rx, ry])

# widgets
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
plt.subplots_adjust(bottom=0.35, top=0.92)

ax_start = plt.axes([0.1, 0.23, 0.15, 0.05]); btn_start = Button(ax_start, "Start")
ax_pause = plt.axes([0.3, 0.23, 0.15, 0.05]); btn_pause = Button(ax_pause, "Pause")
ax_step  = plt.axes([0.5, 0.23, 0.15, 0.05]); btn_step  = Button(ax_step, "Step")
ax_reset = plt.axes([0.7, 0.23, 0.15, 0.05]); btn_reset = Button(ax_reset, "Reset")
ax_reset_all = plt.axes([0.1, 0.15, 0.2, 0.05]); btn_reset_all = Button(ax_reset_all, "Reset All")

ax_q_ball = plt.axes([0.1, 0.08, 0.15, 0.05]); txt_q_ball = TextBox(ax_q_ball, "Ball Q", initial="0.002, 0.05")
ax_r_ball = plt.axes([0.3, 0.08, 0.15, 0.05]); txt_r_ball = TextBox(ax_r_ball, "Ball R", initial=f"{meas_range_noise:.2f}, {meas_bearing_noise:.2f}")

ax_q_lm = plt.axes([0.5, 0.08, 0.15, 0.05]); txt_q_lm = TextBox(ax_q_lm, "LM Q", initial="1e-12")
ax_r_lm = plt.axes([0.7, 0.08, 0.15, 0.05]); txt_r_lm = TextBox(ax_r_lm, "LM R", initial=f"{meas_range_noise:.2f}, {meas_bearing_noise:.2f}")

ax_landmarks = plt.axes([0.65, 0.15, 0.15, 0.05])
txt_landmarks = TextBox(ax_landmarks, "Landmarks", initial=str(n_landmarks))

running = False
paused = False
step_mode = False
current_step = 0

def draw_covariance_ellipse(ax, mean, cov, color='black', alpha=0.25):
    if cov.shape != (2, 2):
        cov = cov[:2, :2]
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor='none', lw=1.5, alpha=alpha)
    ax.add_patch(ell)


def draw_scene():
    ax.clear()
    ax.set_xlim(-1, 11)
    ax.set_ylim(-6, 6)
    ax.set_title("Extended Kalman Filter Simulation")
    ax.set_aspect('equal', adjustable='box')

    if len(robot_path) > 1:
        ax.plot([p[0] for p in robot_path], [p[1] for p in robot_path], '-', color='black', linewidth=1, alpha=0.6)

    ax.plot(robot_gt[0], robot_gt[1], 'ks', markersize=8, label="Robot GT")
    ax.plot(robot_odom[0], robot_odom[1], 'o', color='gray', markersize=5, label="Odom")

    fov_deg = 78
    half_fov = np.deg2rad(fov_deg/2)
    fov_length = 4.0
    theta = robot_gt[2]

    left_x = [robot_gt[0], robot_gt[0] + fov_length * np.cos(theta + half_fov)]
    left_y = [robot_gt[1], robot_gt[1] + fov_length * np.sin(theta + half_fov)]
    right_x = [robot_gt[0], robot_gt[0] + fov_length * np.cos(theta - half_fov)]
    right_y = [robot_gt[1], robot_gt[1] + fov_length * np.sin(theta - half_fov)]
    ax.plot(left_x, left_y, 'b--', alpha=0.8, linewidth=1.2)
    ax.plot(right_x, right_y, 'b--', alpha=0.8, linewidth=1.2)

    for i, lm in enumerate(landmarks):
        ax.scatter(lm[0], lm[1], marker='*', s=120, color=colors[i], label=f"LM{i+1} GT")
        lm_robot = global_to_robot(robot_gt, lm)
        bearing = atan2(lm_robot[1], lm_robot[0])
        ekfs[i].predict(dt)

        visible = abs(wrap_angle(bearing)) <= half_fov

        if visible:
            meas = lm_robot + np.random.randn(2) * meas_range_noise
            meas_global = robot_to_global(robot_odom, meas)

            r_meas = np.linalg.norm(meas)
            b_meas = atan2(meas[1], meas[0])
            z_global = np.array([r_meas, b_meas])

            ekfs[i].update(z_global, robot_odom)
            ax.scatter(meas_global[0], meas_global[1], marker='x', color=colors[i])
            ax.plot([robot_gt[0], meas_global[0]], [robot_gt[1], meas_global[1]],
                linestyle='--', color=colors[i], alpha=0.5)
            ax.text(meas_global[0]+0.1, meas_global[1],
                    f"raw ({meas_global[0]:.2f},{meas_global[1]:.2f})",
                    fontsize=7, color=colors[i])

        ekf_xy = ekfs[i].x
        ax.scatter(ekf_xy[0], ekf_xy[1], marker='o', color=colors[i], ec='black')

        ax.text(ekf_xy[0]+0.1, ekf_xy[1],
                f"filt ({ekf_xy[0]:.2f},{ekf_xy[1]:.2f})",
                fontsize=7, color=colors[i])
        
        draw_covariance_ellipse(ax, ekfs[i].x, ekfs[i].P, color=colors[i], alpha=0.3)
        draw_covariance_ellipse(ax, ekf_ball.x[:2], ekf_ball.P[:2, :2], color='blue', alpha=0.3)

    ax.scatter(ball_pos[0], ball_pos[1], s=100, color='red', marker='o', label="Ball GT")
    ball_robot = global_to_robot(robot_gt, ball_pos)
    meas_ball = ball_robot + np.random.randn(2) * meas_range_noise
    meas_ball_global = robot_to_global(robot_odom, meas_ball)

    bearing = atan2(ball_robot[1], ball_robot[0])
    if abs(wrap_angle(bearing)) > half_fov: return

    r_meas_ball = np.linalg.norm(meas_ball)
    b_meas_ball = atan2(meas_ball[1], meas_ball[0])
    z_ball = np.array([r_meas_ball, b_meas_ball])

    ekf_ball.predict(dt)
    ekf_ball.update(z_ball, robot_odom)

    ax.scatter(meas_ball_global[0], meas_ball_global[1], marker='x', color='red')
    ax.scatter(ekf_ball.x[0], ekf_ball.x[1], marker='o', color='blue')

    ax.plot([robot_gt[0], meas_ball_global[0]], [robot_gt[1], meas_ball_global[1]],
            linestyle='--', color='red', alpha=0.5)

    # teks
    ax.text(meas_ball_global[0]+0.1, meas_ball_global[1],
            f"raw ({meas_ball_global[0]:.2f},{meas_ball_global[1]:.2f})",
            fontsize=7, color='red')
    ax.text(ekf_ball.x[0]+0.1, ekf_ball.x[1],
            f"filt ({ekf_ball.x[0]:.2f},{ekf_ball.x[1]:.2f})",
            fontsize=7, color='blue')

    Pb = ekf_ball.P[:2, :2]
    vals, vecs = np.linalg.eigh(Pb)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2*np.sqrt(vals)
    ax.add_patch(Ellipse(xy=ekf_ball.x[:2], width=width, height=height, angle=angle, alpha=0.2, color='blue'))

    ax.legend(loc="upper left", fontsize=6)

def update_robot_motion():
    global v, omega, robot_gt, robot_odom
    v += np.random.randn()*0.01
    omega += np.random.randn()*0.002
    v = np.clip(v, 0.2, 0.7)
    omega = np.clip(omega, -0.2, 0.2)

    theta = robot_gt[2]
    robot_gt[0] += v * cos(theta) * dt
    robot_gt[1] += v * sin(theta) * dt
    robot_gt[2] += omega * dt

    # field boundary
    if robot_gt[0] <= x_min or robot_gt[0] >= x_max:
        robot_gt[0] = np.clip(robot_gt[0], x_min, x_max)
        robot_gt[2] = np.pi - robot_gt[2]
    if robot_gt[1] <= y_min or robot_gt[1] >= y_max:
        robot_gt[1] = np.clip(robot_gt[1], y_min, y_max)
        robot_gt[2] = -robot_gt[2]

    robot_path.append(robot_gt[:2].copy())

    # odometri noisy
    robot_odom[0] = robot_gt[0] + np.random.randn()*odom_pos_noise
    robot_odom[1] = robot_gt[1] + np.random.randn()*odom_pos_noise
    robot_odom[2] = robot_gt[2] + np.random.randn()*odom_yaw_noise

def update_ball_motion():
    global ball_pos, ball_vel
    ball_pos += ball_vel * dt
    ball_vel += np.random.randn(2) * 0.5
    ball_vel = np.clip(ball_vel, -0.3, 0.3)

# Callback widget
def start(event):
    global running, paused; running, paused = True, False
def pause(event):
    global paused; paused = True
def step_once(event):
    global step_mode; step_mode = True
def reset(event):
    global robot_gt, robot_odom, v, omega, current_step, robot_path, ball_pos, ball_vel
    robot_gt = np.array([0.0, 0.0, 0.0]); robot_odom = robot_gt.copy()
    v, omega = 0.5, 0.05; current_step = 0; robot_path = []
    for i, lm in enumerate(landmarks):
        ekfs[i].x = lm + np.random.randn(2)*2.0; ekfs[i].P = np.eye(2) * 5.0
    ball_pos = np.array([6.0, 0.0]); ball_vel = np.random.randn(2)*0.1
    ekf_ball.x = np.array([6.0, 0.0, 0.0, 0.0]); ekf_ball.P = np.eye(4)*5.0
    draw_scene(); plt.draw()
def reset_all(event):
    global robot_gt, robot_odom, v, omega, current_step, robot_path
    global ball_pos, ball_vel, ekf_ball, landmarks, ekfs, n_landmarks, colors
    try:
        n_landmarks_new = int(txt_landmarks.text)
        if n_landmarks_new > 0: n_landmarks = n_landmarks_new
    except ValueError: pass
    landmarks = np.random.uniform(low=[3.0, -3.0], high=[9.0, 3.0], size=(n_landmarks, 2))
    colors = plt.cm.tab10(np.arange(n_landmarks))
    robot_gt = np.array([0.0, 0.0, 0.0]); robot_odom = robot_gt.copy()
    v, omega = 0.5, 0.05; current_step = 0; robot_path = []
    ekfs = [EKFStatic2D(lm + np.random.randn(2)*2.0, np.eye(2)*5.0) for lm in landmarks]
    ball_pos = np.array([6.0, 0.0]); ball_vel = np.random.randn(2)*0.1
    ekf_ball = EKFBall(np.array([6.0, 0.0, 0.0, 0.0]), np.eye(4)*5.0)
    draw_scene(); plt.draw()

def update_params(text):
    try:
        q_ball = float(txt_q_ball.text); r_ball = float(txt_r_ball.text)
        ekf_ball.Q = np.eye(4) * q_ball; ekf_ball.R = np.eye(2) * r_ball
    except ValueError: pass
    try:
        q_lm = float(txt_q_lm.text); r_lm = float(txt_r_lm.text)
        for ekf in ekfs:
            ekf.Q = np.eye(2) * q_lm; ekf.R = np.eye(2) * r_lm
    except ValueError: pass

txt_q_ball.on_submit(update_params)
txt_r_ball.on_submit(update_params)
txt_q_lm.on_submit(update_params)
txt_r_lm.on_submit(update_params)

btn_start.on_clicked(start); btn_pause.on_clicked(pause)
btn_step.on_clicked(step_once); btn_reset.on_clicked(reset)
btn_reset_all.on_clicked(reset_all)

def on_close(event):
    print("Window ditutup, keluar dari program...")
    plt.close('all')
    sys.exit(0)

fig.canvas.mpl_connect("close_event", on_close)

while True:
    if running and not paused:
        update_robot_motion(); update_ball_motion()
        draw_scene(); current_step += 1; plt.pause(0.05)
    elif step_mode:
        update_robot_motion(); update_ball_motion()
        draw_scene(); current_step += 1; step_mode = False; plt.draw()
    else:
        plt.pause(0.25)
