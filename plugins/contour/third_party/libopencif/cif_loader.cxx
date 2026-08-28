#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "libopencif.hh"

namespace py = pybind11;

namespace {

std::string load_status_name(OpenCIF::File::LoadStatus status)
{
    switch (status) {
    case OpenCIF::File::AllOk:
        return "ok";
    case OpenCIF::File::CantOpenInputFile:
        return "cant_open";
    case OpenCIF::File::IncompleteInputFile:
        return "incomplete";
    case OpenCIF::File::IncorrectInputFile:
        return "incorrect";
    default:
        return "unknown";
    }
}

py::list polygon_points(const OpenCIF::PolygonCommand* command)
{
    py::list points;
    for (const OpenCIF::Point& point : command->getPoints()) {
        points.append(py::make_tuple(point.getX(), point.getY()));
    }
    return points;
}

py::dict load_cif_file(const std::string& path, bool continue_on_error)
{
    OpenCIF::File file;
    file.setPath(path);

    const OpenCIF::File::LoadMethod method =
        continue_on_error ? OpenCIF::File::ContinueOnError : OpenCIF::File::StopOnError;
    const OpenCIF::File::LoadStatus status = file.loadFile(method);

    py::list commands;
    for (OpenCIF::Command* command : file.getCommands()) {
        if (command == nullptr) {
            continue;
        }

        switch (command->type()) {
        case OpenCIF::Command::Polygon: {
            const auto* polygon = static_cast<OpenCIF::PolygonCommand*>(command);
            py::dict item;
            item["type"] = "polygon";
            item["points"] = polygon_points(polygon);
            commands.append(item);
            break;
        }
        case OpenCIF::Command::Box: {
            const auto* box = static_cast<OpenCIF::BoxCommand*>(command);
            const OpenCIF::Size size = box->getSize();
            const OpenCIF::Point position = box->getPosition();
            const OpenCIF::Point rotation = box->getRotation();
            py::dict item;
            item["type"] = "box";
            item["width"] = static_cast<long long>(size.getWidth());
            item["height"] = static_cast<long long>(size.getHeight());
            item["center_x"] = position.getX();
            item["center_y"] = position.getY();
            item["rotation_x"] = rotation.getX();
            item["rotation_y"] = rotation.getY();
            commands.append(item);
            break;
        }
        case OpenCIF::Command::Comment: {
            const auto* comment = static_cast<OpenCIF::CommentCommand*>(command);
            py::dict item;
            item["type"] = "comment";
            item["content"] = comment->getContent();
            commands.append(item);
            break;
        }
        default:
            break;
        }
    }

    py::dict result;
    result["status"] = load_status_name(status);
    result["messages"] = file.getMessages();
    result["commands"] = commands;
    return result;
}

} // namespace

PYBIND11_MODULE(cif_loader, m)
{
    m.doc() = "LibOpenCIF bindings for Contour CIF loading";
    m.def("load_cif_file", &load_cif_file, py::arg("path"), py::arg("continue_on_error") = true);
    m.def("library_version", []() { return OpenCIF::LibraryVersion; });
    m.def("library_name", []() { return OpenCIF::LibraryName; });
}
