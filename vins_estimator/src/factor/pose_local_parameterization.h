#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"
/**
 * @brief 自己定义的加法,需要实现Plus,ComputeJacobian,GlobalSize,LocalSize
 *
 */
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; }; //参数个数(t:3个, R:4个(四元数))
    virtual int LocalSize() const { return 6; }; //实际自由度(t:3个, R:3个)
};
