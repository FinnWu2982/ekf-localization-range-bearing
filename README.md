# EKF Localization with Range-Bearing Landmarks

This project implements a 2D robot localization system using an Extended Kalman Filter (EKF) with odometry and range-bearing landmark measurements.

## Overview
The robot is modeled with a unicycle motion model. Landmark observations are given as range and bearing measurements with a forward sensor offset. The estimator fuses noisy odometry and exteroceptive measurements to estimate the robot pose over time.

## Features
- Extended Kalman filter for nonlinear localization
- Range-bearing landmark observation model
- Support for different sensing ranges
- Experiments with poor initialization
- Joseph-form covariance update for improved numerical stability
- Trajectory animation with covariance ellipse visualization

## File
- `ekf_joseph.py`: main EKF implementation

## Requirements
- numpy
- scipy
- matplotlib

## Notes
This repository is adapted from a graduate state estimation course project. The current public version keeps the Joseph-form covariance update for better numerical stability.
