#define _USE_MATH_DEFINES
#include "Eigen/Dense"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using std::cout;
using std::endl;

// normalize angle to [-pi, pi]
double wrapAngle(double a) {
    while (a > M_PI) a -= 2*M_PI;
    while (a <= -M_PI) a += 2*M_PI;
    return a;
}

// structure for robot pose (known)
struct Pose {
    double x, y, theta;
};

// measurement: range and bearing
struct Measurement {
    double range;
    double bearing;
};

// generate synthetic robot poses (circle or line)
std::vector<Pose> generateRobotPoses(int n) {
    std::vector<Pose> poses;
    poses.reserve(n);
    // example: robot moves along x axis while rotating slowly
    for (int i = 0; i < n; ++i) {
        double t = double(i) * 0.2;
        Pose p;
        p.x = -2.0 + 0.5 * i;          // move along x
        p.y = 0.15 * std::sin(i * 0.15); // small y wiggle
        p.theta = 0.02 * i;             // slowly turning
        poses.push_back(p);
    }
    return poses;
}

// compute true measurement for landmark position (lx, ly) and pose
Measurement computeTrueMeasurement(const Pose &p, const Vector2d &landmark) {
    double dx = landmark(0) - p.x;
    double dy = landmark(1) - p.y;
    double r = std::sqrt(dx*dx + dy*dy);
    double bearing = std::atan2(dy, dx) - p.theta;
    bearing = wrapAngle(bearing);
    return {r, bearing};
}

// add Gaussian noise to measurement
Measurement addNoise(const Measurement &m, std::mt19937 &rng, double sigma_r, double sigma_b) {
    std::normal_distribution<double> nr(0.0, sigma_r);
    std::normal_distribution<double> nb(0.0, sigma_b);
    return { m.range + nr(rng), wrapAngle(m.bearing + nb(rng)) };
}

// EKF update for static landmark (state: [lx, ly])
// x: 2x1, P: 2x2, z: measurement (range, bearing), pose: robot pose, R: 2x2 measurement covariance
void ekfUpdate(Vector2d &x, Matrix2d &P, const Measurement &z, const Pose &p, const Matrix2d &R) {
    // h(x)
    double dx = x(0) - p.x;
    double dy = x(1) - p.y;
    double q = dx*dx + dy*dy;
    if(q < 1e-8) q = 1e-8;
    double pred_r = std::sqrt(q);
    double pred_b = std::atan2(dy, dx) - p.theta;
    pred_b = wrapAngle(pred_b);

    // Jacobian H (2x2) of h with respect to landmark state (lx, ly)
    // h = [sqrt(q); atan2(dy,dx) - theta]
    // dh/dlx = [dx/r, -dy/q? No; be careful]
    // derivative:
    // dr/dlx = dx / r
    // dr/dly = dy / r
    // db/dlx = -dy / q
    // db/dly = dx / q
    Matrix<double, 2, 2> H;
    if (pred_r < 1e-8) pred_r = 1e-8;
    H(0,0) = dx / pred_r;
    H(0,1) = dy / pred_r;
    H(1,0) = -dy / q;
    H(1,1) = dx / q;

    // Innovation y = z - h(x)
    Vector2d zvec;
    zvec(0) = z.range;
    zvec(1) = z.bearing;
    Vector2d hvec;
    hvec(0) = pred_r;
    hvec(1) = pred_b;
    Vector2d y = zvec - hvec;
    y(1) = wrapAngle(y(1));

    // S = H P H^T + R
    Matrix2d S = H * P * H.transpose() + R;

    // K = P H^T S^-1
    Matrix<double, 2, 2> K = P * H.transpose() * S.inverse();

    // update state
    x = x + K * y;

    // update covariance
    Matrix2d I = Matrix2d::Identity();
    P = (I - K * H) * P;
}

// compute RMSE between estimated and true
double rmse(const Vector2d &est, const Vector2d &truth) {
    Vector2d e = est - truth;
    return std::sqrt(e.squaredNorm());
}

int main() {
    // random engine
    std::random_device rd;
    std::mt19937 rng(rd());

    // 1) True landmark position (unknown to filter)
    Vector2d landmark_true;
    landmark_true << 1.5, 0.8;

    // 2) Generate robot poses and noisy measurements
    int N = 50;
    auto poses = generateRobotPoses(N);

    // measurement noise stddev
    double sigma_range = 0.05;    // meters
    double sigma_bearing = 0.03;  // radians (~1.7 deg)

    // measurement covariance R
    Matrix2d R = Matrix2d::Zero();
    R(0,0) = sigma_range * sigma_range;
    R(1,1) = sigma_bearing * sigma_bearing;

    // Create noisy measurements vector
    std::vector<Measurement> measurements;
    measurements.reserve(N);
    for (int i = 0; i < N; ++i) {
        Measurement true_z = computeTrueMeasurement(poses[i], landmark_true);
        Measurement noisy = addNoise(true_z, rng, sigma_range, sigma_bearing);
        measurements.push_back(noisy);
    }

    // 3) EKF initialization
    Vector2d x_est;         // initial guess for landmark position
    x_est << 0.0, 0.0;      // deliberately far from true
    Matrix2d P = Matrix2d::Identity() * 5.0; // large uncertainty

    // Optional process noise for static landmark (small)
    Matrix2d Q = Matrix2d::Identity() * 1e-6;

    cout << std::fixed << std::setprecision(4);
    cout << "True landmark: [" << landmark_true(0) << ", " << landmark_true(1) << "]\n";
    cout << "Initial estimate: [" << x_est(0) << ", " << x_est(1) << "], P=\n" << P << "\n\n";

    // 4) Loop through measurements and update EKF
    for (int i = 0; i < N; ++i) {
        // Prediction for static landmark: x = x, P = P + Q
        P = P + Q;

        // Update with measurement i
        ekfUpdate(x_est, P, measurements[i], poses[i], R);

        // Print progress occasionally
        if ((i % 5) == 0 || i == N-1) {
            double err = rmse(x_est, landmark_true);
            cout << "Step " << i+1 << " | est = [" << x_est(0) << ", " << x_est(1) << "]"
                 << " | RMSE = " << err << " | P_diag = [" << P(0,0) << ", " << P(1,1) << "]\n";
        }
    }

    // final results
    double final_rmse = rmse(x_est, landmark_true);
    cout << "\nFinal estimate: [" << x_est(0) << ", " << x_est(1) << "]\n";
    cout << "Final covariance P =\n" << P << "\n";
    cout << "Final RMSE = " << final_rmse << "\n";

    // show a few raw measurements vs true
    cout << "\nSample of first 6 measurements (range, bearing) [noisy] vs true:\n";
    for (int i = 0; i < std::min(6, N); ++i) {
        auto t = computeTrueMeasurement(poses[i], landmark_true);
        auto m = measurements[i];
        cout << "i=" << i
             << " noisy=(r=" << m.range << ", b=" << m.bearing << ")"
             << " true=(r=" << t.range << ", b=" << t.bearing << ")\n";
    }

    return 0;
}
