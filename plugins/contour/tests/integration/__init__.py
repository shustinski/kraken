# Golden snapshot of PolygonExtractionWidget public API.
# Regenerate via tests/unit/test_widget_smoke.py::regenerate_snapshot() when a
# change is intentional; otherwise any diff is a regression.
#
# Format: one SIGNAL/METHOD name per line. Inherited Qt members are excluded.

[signals]
batchFinished
batchProgress
imageProcessed
logMessage
polygonsEdited

[methods]
attach_help_menu
export_current_frame_to_dataset
get_pipeline
get_polygons
help_menu_title
load_image
load_images
process_current_image
refresh_image_list
save_current_result
set_cif_directory
set_dataset_directory
set_input_directory
set_output_directory
set_pipeline
set_ui_language
start_batch_processing
stop_batch_processing
