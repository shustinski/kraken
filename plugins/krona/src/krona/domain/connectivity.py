"""Electrical identities declared by EDIF joined/portRef, independent of drawing.

Multiple net declarations for the same pin are an input conflict. They are not
silently merged into a new net. A drawing endpoint never adds a connection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .netlist import TopLevelNetlist

POWER_PORT_NAMES = frozenset({"1", "VDD", "VCC", "+5", "5PLUS", "POWER"})
GROUND_PORT_NAMES = frozenset({"0", "&0", "GND", "VSS"})


@dataclass(frozen=True)
class ExplicitConnectivity:
    pins: dict[str, dict[str, str]]
    top_ports: dict[str, str]
    directions: dict[str, str]
    issues: tuple[dict, ...]
    invalid_instances: frozenset[str]

    @classmethod
    def from_netlist(cls, netlist: TopLevelNetlist):
        issues = []
        counts = Counter(net.name for net in netlist.nets)
        duplicate_nets = {name for name, count in counts.items() if count > 1}
        for name in sorted(duplicate_nets):
            issues.append({"code": "duplicate_net", "severity": "error", "net": name})
        instance_counts = Counter(instance.name for instance in netlist.instances)
        invalid = {name for name, count in instance_counts.items() if count > 1}
        for name in sorted(invalid):
            issues.append({"code": "duplicate_instance", "severity": "error", "instance": name})
        endpoints = defaultdict(set)
        transistor_instances = {
            instance.name for instance in netlist.instances
            if any(token in " ".join(filter(None, [instance.name, instance.cell, instance.view, instance.library])).upper()
                   for token in ("NMOS", "PMOS"))
        }
        for net in netlist.nets:
            if not net.connections:
                issues.append({"code": "no_explicit_connections", "severity": "warning", "net": net.name})
            for connection in net.connections:
                if connection.instance is not None and connection.instance not in instance_counts:
                    issues.append({"code": "unknown_instance", "severity": "error", "instance": connection.instance,
                                   "port": connection.port, "net": net.name})
                    continue
                port = connection.port.upper()
                if connection.instance in transistor_instances:
                    port = {"G": "GATE", "S": "SOURCE", "D": "DRAIN"}.get(port, port)
                key = (connection.instance, port)
                endpoints[key].add(net.name)
                if net.name in duplicate_nets and connection.instance:
                    invalid.add(connection.instance)
        pins = defaultdict(dict)
        top_ports = {}
        for (instance, port), names in sorted(endpoints.items(), key=lambda item: (item[0][0] or "", item[0][1])):
            if len(names) != 1 or names & duplicate_nets:
                issues.append({"code": "conflicting_port_nets", "severity": "error", "instance": instance,
                               "port": port, "nets": sorted(names)})
                if instance:
                    invalid.add(instance)
                continue
            name = next(iter(names))
            if instance:
                pins[instance][port] = name
            else:
                top_ports[port] = name
        directions = {port.name.upper(): port.direction for port in netlist.ports if port.direction}
        for port, direction in directions.items():
            if direction not in {"input", "output", "inout"}:
                issues.append({"code": "unsupported_port_direction", "severity": "error", "port": port})
        # Different declarations of the same interface port do not justify
        # choosing one direction by declaration order.
        for name, count in Counter(port.name.upper() for port in netlist.ports).items():
            if count > 1:
                directions.pop(name, None)
                top_ports.pop(name, None)
                issues.append({"code": "duplicate_interface_port", "severity": "error", "port": name})
        for instance in sorted(transistor_instances):
            missing = {"GATE", "SOURCE", "DRAIN"} - pins.get(instance, {}).keys()
            if missing:
                invalid.add(instance)
                issues.append({"code": "missing_explicit_terminals", "severity": "error",
                               "instance": instance, "ports": sorted(missing)})
        power = {net.name for net in netlist.nets if net.name.upper() in POWER_PORT_NAMES}
        ground = {net.name for net in netlist.nets if net.name.upper() in GROUND_PORT_NAMES}
        power.update(net for port, net in top_ports.items() if port in POWER_PORT_NAMES)
        ground.update(net for port, net in top_ports.items() if port in GROUND_PORT_NAMES)
        for net in sorted(power & ground):
            issues.append({"code": "conflicting_rail_roles", "severity": "error", "net": net})
        return cls(dict(pins), top_ports, directions, tuple(issues), frozenset(invalid))

    def ports_with_direction(self, direction):
        return {port: net for port, net in self.top_ports.items() if self.directions.get(port) == direction}

    def provenance(self):
        return {"instances": self.pins, "top_ports": self.top_ports, "directions": self.directions,
                "issues": list(self.issues), "invalid_instances": sorted(self.invalid_instances)}
