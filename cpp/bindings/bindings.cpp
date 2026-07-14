#include "cloth_opt/cloth.h"
#include "cloth_opt/controller.h"
#include "cloth_opt/integrator.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
using namespace ClothOpt;

namespace {

Eigen::MatrixXd vectorsToMatrix(const std::vector<Eigen::Vector3d>& values) {
    Eigen::MatrixXd result(values.size(), 3);
    for (size_t i = 0; i < values.size(); ++i) {
        result.row(i) = values[i].transpose();
    }
    return result;
}

void matrixToVectors(const Eigen::Ref<const Eigen::MatrixXd>& values,
                     std::vector<Eigen::Vector3d>& destination) {
    if (values.cols() != 3 || static_cast<size_t>(values.rows()) != destination.size()) {
        throw std::invalid_argument("expected shape (vertex_count, 3)");
    }
    for (Eigen::Index i = 0; i < values.rows(); ++i) {
        destination[i] = values.row(i).transpose();
    }
}

void setPositions(ClothMesh& mesh, const Eigen::Ref<const Eigen::MatrixXd>& values) {
    if (values.cols() != 3 || static_cast<size_t>(values.rows()) != mesh.getVertexCount()) {
        throw std::invalid_argument("expected shape (vertex_count, 3)");
    }
    for (Eigen::Index i = 0; i < values.rows(); ++i) {
        mesh.setVertexPosition(static_cast<size_t>(i), values.row(i).transpose());
    }
}

void setPinned(ClothMesh& mesh, const std::vector<bool>& values) {
    if (values.size() != mesh.getVertexCount()) {
        throw std::invalid_argument("expected one pin flag per vertex");
    }
    mesh.pinned = values;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i]) mesh.velocities[i] = Eigen::Vector3d::Zero();
    }
}

void simulate(ClothMesh& mesh, ClothController& controller,
              SemiImplicitEulerIntegrator& integrator, double dt, int substeps) {
    if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
    if (substeps < 0) throw std::invalid_argument("substeps must be non-negative");
    for (int i = 0; i < substeps; ++i) {
        controller.applyControls(mesh, dt);
        integrator.step(mesh, dt);
    }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Bindings for the existing ClothOpt C++ simulation core";

    py::class_<ClothProperties>(m, "ClothProperties")
        .def(py::init<>())
        .def_readwrite("mass", &ClothProperties::mass)
        .def_readwrite("stiffness", &ClothProperties::stiffness)
        .def_readwrite("damping", &ClothProperties::damping)
        .def_readwrite("bending_stiffness", &ClothProperties::bendingStiffness)
        .def_readwrite("friction", &ClothProperties::friction)
        .def_readwrite("gravity", &ClothProperties::gravity);

    py::class_<DebugInfo>(m, "DebugInfo")
        .def_readonly("total_force", &DebugInfo::totalForce)
        .def_readonly("total_velocity", &DebugInfo::totalVelocity)
        .def_readonly("max_displacement", &DebugInfo::maxDisplacement)
        .def_readonly("active_vertices", &DebugInfo::activeVertices)
        .def_readonly("pinned_vertices", &DebugInfo::pinnedVertices)
        .def_readonly("gravity_magnitude", &DebugInfo::gravityMagnitude)
        .def_readonly("spring_force_magnitude", &DebugInfo::springForceMagnitude);

    py::class_<ClothMesh>(m, "ClothMesh")
        .def(py::init<>())
        .def("create_grid", &ClothMesh::createGrid, py::arg("width"), py::arg("height"),
             py::arg("spacing") = 0.1)
        .def("pin_vertex", &ClothMesh::pinVertex)
        .def("pin_corners", &ClothMesh::pinCorners)
        .def("add_sphere", &ClothMesh::addSphere)
        .def("clear_spheres", &ClothMesh::clearSpheres)
        .def("grid_index", &ClothMesh::getGridIndex)
        .def_property_readonly("vertex_count", &ClothMesh::getVertexCount)
        .def_property_readonly("triangle_count", &ClothMesh::getTriangleCount)
        .def_property("positions", &ClothMesh::getVertexMatrix, &setPositions)
        .def_property("velocities",
                      [](const ClothMesh& self) { return vectorsToMatrix(self.velocities); },
                      [](ClothMesh& self, const Eigen::Ref<const Eigen::MatrixXd>& value) {
                          matrixToVectors(value, self.velocities);
                      })
        .def_property("pinned",
                      [](const ClothMesh& self) { return self.pinned; }, &setPinned)
        .def_property_readonly("triangles", &ClothMesh::getTriangleMatrix)
        .def_readwrite("properties", &ClothMesh::properties);

    py::class_<SemiImplicitEulerIntegrator>(m, "SemiImplicitEulerIntegrator")
        .def(py::init<>())
        .def("step", &SemiImplicitEulerIntegrator::step)
        .def("enable_debug", &SemiImplicitEulerIntegrator::enableDebug)
        .def("enable_verbose_debug", &SemiImplicitEulerIntegrator::enableVerboseDebug)
        .def("set_debug_frequency", &SemiImplicitEulerIntegrator::setDebugFrequency)
        .def_property_readonly("last_debug_info", &SemiImplicitEulerIntegrator::getLastDebugInfo,
                               py::return_value_policy::reference_internal);

    py::class_<ClothController>(m, "ClothController")
        .def(py::init<>())
        .def("add_position_control", &ClothController::addPositionControl,
             py::arg("vertex_index"), py::arg("target_position"), py::arg("gain") = 1000.0,
             py::arg("max_force") = 100.0)
        .def("add_velocity_control", &ClothController::addVelocityControl,
             py::arg("vertex_index"), py::arg("target_velocity"), py::arg("gain") = 100.0,
             py::arg("max_force") = 50.0)
        .def("add_force_control", &ClothController::addForceControl)
        .def("update_position_target", &ClothController::updatePositionTarget)
        .def("update_velocity_target", &ClothController::updateVelocityTarget)
        .def("update_force_target", &ClothController::updateForceTarget)
        .def("remove_control", &ClothController::removeControl)
        .def("remove_all_controls", &ClothController::removeAllControls)
        .def("apply_controls", &ClothController::applyControls)
        .def("set_trajectory", &ClothController::setTrajectory,
             py::arg("vertex_index"), py::arg("positions"), py::arg("times"),
             py::arg("loop") = false)
        .def("add_circular_motion", &ClothController::addCircularMotion,
             py::arg("vertex_index"), py::arg("center"), py::arg("radius"),
             py::arg("frequency"), py::arg("axis") = Eigen::Vector3d(0, 1, 0))
        .def("add_sinusoidal_motion", &ClothController::addSinusoidalMotion)
        .def("add_wind_force", &ClothController::addWindForce,
             py::arg("vertex_indices"), py::arg("wind_direction"), py::arg("strength"),
             py::arg("turbulence") = 0.0)
        .def("is_controlled", &ClothController::isControlled)
        .def("get_control_force", &ClothController::getControlForce)
        .def_property_readonly("controlled_vertices", &ClothController::getControlledVertices)
        .def_property_readonly("control_count", &ClothController::getControlCount)
        .def("enable_debug", &ClothController::enableDebug);

    m.def("simulate", &simulate, py::arg("mesh"), py::arg("controller"),
          py::arg("integrator"), py::arg("dt"), py::arg("substeps"),
          py::call_guard<py::gil_scoped_release>());
}
