#define _USE_MATH_DEFINES
#include "Eigen/Dense"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace Eigen;
using std::cout;

double wrapAngle(double a) {
    while (a > M_PI) a -= 2*M_PI;
    while (a <= -M_PI) a += 2*M_PI;
    return a;
}

struct Pose {
    double x, y, theta;
};

struct Measurement {
    double range;
    double bearing;
};

std::vector<Pose> generateRobotPoses(int n) {
    std::vector<Pose> poses;
    poses.reserve(n);
    for (int i = 0; i < n; i++) {
        double t = double(i)*0.2;
        Pose p;
        p.x = -2.00 + 0.35*i;      
        p.y = 0.1*i;         
        p.theta = 0.02*i;    
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

Measurement addNoise(const Measurement &m, std::mt19937 &rng, double sigma_r, double sigma_b) {
    std::normal_distribution<double> nr(0.0, sigma_r);
    std::normal_distribution<double> nb(0.0, sigma_b);
    return {m.range + nr(rng), wrapAngle(m.bearing + nb(rng))};
}

void ekfUpdate(Vector2d &x, Matrix2d &P, const Measurement &z, const Pose &p, const Matrix2d &R) {
    double dx = x(0) - p.x;
    double dy = x(1) - p.y;
    double q = dx*dx + dy*dy;
    if(q < 1e-8) q = 1e-8;
    double pred_r = std::sqrt(q);
    double pred_b = std::atan2(dy, dx) - p.theta;
    pred_b = wrapAngle(pred_b);

    Matrix<double, 2, 2> H;
    if (pred_r < 1e-8) pred_r = 1e-8;
    H(0,0) = dx / pred_r;
    H(0,1) = dy / pred_r;
    H(1,0) = -dy / q;
    H(1,1) = dx / q;

    Vector2d zvec;
    zvec(0) = z.range;
    zvec(1) = z.bearing;
    Vector2d hvec;
    hvec(0) = pred_r;
    hvec(1) = pred_b;
    Vector2d y = zvec - hvec;
    y(1) = wrapAngle(y(1));

    Matrix2d S = H*P*H.transpose() + R;

    Matrix<double, 2, 2> K = P*H.transpose()*S.inverse();

    x = x + K*y;

    Matrix2d I = Matrix2d::Identity();
    P = (I - K*H)*P;
}

double rmse(const Vector2d &est, const Vector2d &truth) {
    Vector2d e = est - truth;
    return std::sqrt(e.squaredNorm());
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    Vector2d landmark_true;
    landmark_true << 1.5, 0.8;

    int N = 50;
    auto poses = generateRobotPoses(N);
    double sigma_range = 0.1;  
    double sigma_bearing = 0.05; 

    Matrix2d R = Matrix2d::Zero();
    R(0,0) = sigma_range*sigma_range;
    R(1,1) = sigma_bearing*sigma_bearing;

    std::vector<Measurement> measurements;
    measurements.reserve(N);
    for (int i = 0; i < N; i++) {
        Measurement true_z = computeTrueMeasurement(poses[i], landmark_true);
        Measurement noisy = addNoise(true_z, rng, sigma_range, sigma_bearing);
        measurements.push_back(noisy);
    }
    Vector2d x_est;       
    x_est << 0.0, 0.0;    
    Matrix2d P = Matrix2d::Identity()*5.0; //mengatur confidence score
    Matrix2d Q = Matrix2d::Identity()*1e-6;

    cout << std::fixed << std::setprecision(4);
    cout << "True landmark: [" << landmark_true(0) << ", " << landmark_true(1) << "]\n";
    cout << "Initial estimate: [" << x_est(0) << ", " << x_est(1) << "],\nP=\n" << P << "\n\n";

    for (int i = 0; i < N; i++) {
        P = P+Q;
        ekfUpdate(x_est, P, measurements[i], poses[i], R);

        if ((i % 4) == 0 || i == N-1) {
            double err = rmse(x_est, landmark_true);
            cout << "Step " << i+1 << " | estimate = [" << x_est(0) << ", " << x_est(1) << "]"
                << " | RMSE = " << err << " | Covar_matrix = [" << P(0,0) << ", " << P(1,1) << "]\n";
        }
    }

    double final_rmse = rmse(x_est, landmark_true);
    cout << "\nFinal estimate: [" << x_est(0) << ", " << x_est(1) << "]\n";
    cout << "Final covariance P =\n" << P << "\n";
    cout << "Final RMSE = " << final_rmse << "\n";

    cout << "\nSample of first 10 measurements (range, bearing) [noisy] vs true:\n";
    for (int i = 0; i < 10; i++) {
        auto t = computeTrueMeasurement(poses[i], landmark_true);
        auto m = measurements[i];
        cout << "i=" << i+1
            << " noisy=(r=" << m.range << ", b=" << m.bearing << ")"
            << " true=(r=" << t.range << ", b=" << t.bearing << ")\n";
    }

    return 0;
}
