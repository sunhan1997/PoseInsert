#include <mutex>
#include <vector>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry> // 包含变换和位姿处理

namespace arx::r5
{
    class InterfacesTools
    {
    public:
        InterfacesTools(int flag);
        std::vector<double> ForwardKinematicsRpy(std::vector<double>);

    private:
        class impl;
        std::unique_ptr<impl> pimpl;
    };
}
