# EKF Landmark & Ball Tracking Visualization 🧠
This project implements **Extended Kalman Filter (EKF)** for estimating and visualizing landmark positions and a dynamic ball in a simulated 2D Environment. The simulation includes random robot poses, noisy range-bearing measurements, and matplotlib visualization.


## Features 🚀
- Visualization of the robot, landmarks, and a dynamic ball.
- EKF update for both **landmarks** and **ball position estimation**
- Mahalanobis distance calculation test for data association
- Configurable noise parameters and covariance tuning
- Step-by-step simulation playback using matplotlib


## Project Structure 🧩
ekf-visualization.py # main simulation script

ekf-implement.cpp # ekf implementation for single landmark in C++

README.md # project description (this file)


## How It Works ⚙️
1. The robot pose `(x, y, θ)` is randomized at each step
2. The environment contains multiple known landmarks with one dynamic ball
3. The robot sense each visible object within its Field of View (FOV) using range and bearing
4. Each measurement is corrupted with Gaussian Noise :
   - Range noise
   - Bearing noise
5. The EKF function performs :
   - Measurement prediction from the current estimate
   - Kalman gain update
   - Covariance propagation
6. The results are visualized dynamically with `matplotlib`.


## Tunable parameters 🔢
sigma_r = 0.30           # range measurement noise (m)

sigma_b = 1.00           # bearing noise (rad)

P_init = np.eye(2)*3.2   # initial landmark covariance

Q_lm = np.eye(2)*1e-5    # process noise covariance

# Requirements 💻
**python 3 installed**

pip install numpy matplotlib pandas
