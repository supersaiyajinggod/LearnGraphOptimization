# LearnGraphOptimization

## Requirements

-	Eigen3	<https://eigen.tuxfaily.org>

On Ubuntu / Debian these dependencies are resolved by installing the
following packages.
-	libeigen3-dev

## Examples

There are two examples:
1. Fitting the curve: y  = exp(a * x * x + b * x + c)
2. Calculate a camera pose.

## Complation
Just build each instance in the folder directly.

### Build CurveFitting
-	`cd examples/CurveFitting`
-	`mkdir build`
-	`cmake ..`
-	`make`

### Build SLAM
-	`cd examples/SLAM`
-	`mkdir build`
-	`cmake ..`
-	`make`
