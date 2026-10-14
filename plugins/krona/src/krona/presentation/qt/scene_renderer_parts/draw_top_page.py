"""scene_renderer: draw top page."""
from __future__ import annotations
from krona.application.ports import SceneData
from krona.infrastructure.edif_parser import EdifTextParser

from krona.presentation.qt.scene_renderer_parts.affine2_d import Affine2D


class DrawTopPageMixin:
    def _draw_top_page(self, scene_data: SceneData) -> None:
        for child in EdifTextParser.direct_children(scene_data.top_page_block):
            stripped = child.lstrip()
            if stripped.startswith("(net "):
                net_block = child
                for figure_block in EdifTextParser.find_direct_classes(net_block, "(figure "):
                    wrapper = f"(tmp {figure_block})"
                    self._draw_container(wrapper, Affine2D(), scene_data.netlist.library, scene_data)
            elif stripped.startswith("(portImplementation"):
                # Top-level page ports (e.g., OUTx) are represented as portImplementation.
                self._draw_port_implementation(child, (180, 30, 30))
            elif stripped.startswith("(instance "):
                instance_block = child
                if self._is_hidden_instance(instance_block):
                    continue
                inst_transform = self._transform_from_block(instance_block)
                view_ref = EdifTextParser.first_direct_or_none(instance_block, "(viewRef ")
                cell_ref = EdifTextParser.first_any_or_none(instance_block, "(cellRef ")
                lib_ref = EdifTextParser.first_any_or_none(instance_block, "(libraryRef ")
                if not view_ref or not cell_ref or not lib_ref:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                view_name = EdifTextParser.parse_header_name(view_ref, "viewRef")
                cell_name = EdifTextParser.parse_header_name(cell_ref, "cellRef")
                lib_name = EdifTextParser.parse_header_name(lib_ref, "libraryRef")
                view_block = scene_data.view_index.get((lib_name, cell_name, view_name))
                if not view_block:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                draw_block = self._resolve_view_draw_block(view_block)
                if not draw_block:
                    self._draw_instance_overlay_fallback(instance_block, inst_transform)
                    continue
                instance_name = EdifTextParser.parse_header_name(instance_block, "instance")
                instance_color = self._resolve_instance_primary_color(draw_block, lib_name)
                self._draw_container(
                    draw_block,
                    inst_transform,
                    lib_name,
                    scene_data,
                    click_owner_instance=instance_name,
                )
                if self._dark_mode and self._is_mos_instance(instance_block):
                    self._draw_mos_contrast_overlay(draw_block, inst_transform)
                self._draw_port_instances(instance_block, inst_transform, instance_color, draw_block)
                self._draw_pmos_gate_bubble(instance_block, inst_transform, draw_block, instance_color)
                self._draw_instance_hitbox(instance_block, inst_transform, draw_block)
                if not self._is_port_instance_for_visualization(instance_block):
                    self._draw_string_displays(
                        instance_block,
                        Affine2D(),
                        scene_data.netlist.library,
                        text_scale=0.1,
                    )
