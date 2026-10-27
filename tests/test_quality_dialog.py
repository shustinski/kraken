from tests.test_matrix_lifetime import run_qt_child


def test_quality_dialog_drives_real_local_issue_assignment_and_history():
    run_qt_child("""
        import tempfile, time
        from pathlib import Path
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtTest import QTest
        from kraken_hub.quality_dialog import QualityDialog
        from tests.test_work_orders import _service, _project_with_frames
        app = QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = _service(root)
            owner, performer, project, layer, images, vectors = _project_with_frames(service, root/'source', with_vectors=False)
            workers = set()
            dialog = QualityDialog(service, owner.principal, project.id, background_workers=workers)
            dialog.show()
            def idle():
                deadline = time.monotonic()+8
                while time.monotonic()<deadline:
                    QTest.qWait(20)
                    if dialog._worker is None and not dialog._deferred_details:
                        QTest.qWait(30)
                        if dialog._worker is None:
                            return
                raise AssertionError(dialog.status.text())
            idle()
            dialog.title.setText('Conductor gap')
            dialog.description.setText('Gap in selected frame')
            dialog.load_frame_versions()
            idle()
            assert dialog.versions.count() == 1
            dialog.versions.setCurrentRow(0)
            dialog.create_issue()
            idle()
            assert dialog.issue.state.value == 'open', dialog.status.text()
            assert dialog.table.rowCount() == 1
            dialog.note.setText('Repair assigned')
            dialog.command('assign')
            idle()
            assert dialog.issue.state.value == 'assigned', dialog.status.text()
            assert 'Repair assigned' in dialog.history.toPlainText()
            dialog.note.setText('Correction cannot reuse the source version')
            dialog.versions.setCurrentRow(0)
            dialog.command('correct')
            idle()
            assert dialog.issue.state.value == 'assigned'
            assert dialog.status.text().startswith('Ошибка:')
            dialog.refresh()
            idle()
            assert dialog.issue.revision == 1
            dialog.close()
            assert not workers
            del dialog
            app.processEvents()
    """)


def test_quality_dialog_analysis_rule_uses_measured_runs_without_default_threshold():
    run_qt_child("""
        import sys, tempfile, time
        from pathlib import Path
        sys.path.insert(0, str(Path('plugins/karakal/src').resolve()))
        import pytest
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtTest import QTest
        from kraken_hub.quality_dialog import QualityDialog
        from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
        from tests.test_quality_workflow import quality
        from tests.test_quality_analysis import real_inputs, real_run
        app = QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q = quality.__wrapped__(root, pytest.MonkeyPatch())
            target, reference, correction = real_inputs(q, root)
            store = FilesystemAnalysisStore(q.service.catalog_root, str(q.project.id))
            original, _ = real_run(q, root/'original', store, target, reference)
            checked, _ = real_run(q, root/'checked', store, correction, reference)
            workers = set()
            dialog = QualityDialog(q.service, q.owner, q.project.id, background_workers=workers)
            dialog.show()
            def idle():
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    QTest.qWait(20)
                    if dialog._worker is None and not dialog._deferred_details:
                        QTest.qWait(30)
                        if dialog._worker is None: return
                raise AssertionError(dialog.status.text())
            idle()
            dialog.load_analysis_runs()
            idle()
            assert dialog.analysis_runs.count()==2
            for index in range(dialog.analysis_runs.count()):
                if dialog.analysis_runs.itemData(index)['run_id']==original.run_id:
                    dialog.analysis_runs.setCurrentIndex(index)
                    break
            assert not dialog.analysis_threshold.text()
            dialog.create_analysis_findings()
            assert 'порог' in dialog.status.text()
            assert q.service.list_quality_issues(q.project.id)==()
            dialog.analysis_threshold.setText('0.9')
            dialog.create_analysis_findings()
            idle()
            assert dialog.issue.analysis_run_id==original.run_id, dialog.status.text()
            assert dialog.issue.analysis_rule.metric_key=='iou'
            dialog.note.setText('Repair measured target')
            dialog.command('assign')
            idle()
            for index in range(dialog.versions.count()):
                if dialog.versions.item(index).data(Qt.ItemDataRole.UserRole)==str(correction.id):
                    dialog.versions.setCurrentRow(index)
                    break
            dialog.note.setText('Target mask corrected')
            dialog.command('correct')
            idle()
            assert dialog.issue.state.value=='corrected', dialog.status.text()
            dialog.principal=q.reviewer
            for index in range(dialog.analysis_runs.count()):
                if dialog.analysis_runs.itemData(index)['run_id']==checked.run_id:
                    dialog.analysis_runs.setCurrentIndex(index)
                    break
            dialog.note.setText('Independent measured recheck')
            dialog.recheck_analysis_finding()
            idle()
            assert dialog.issue.reviews[-1].passed, dialog.status.text()
            assert dialog.issue.reviews[-1].observations[0][1]==1.0
            dialog.note.setText('Accepted measured repair')
            dialog.command('accept')
            idle()
            assert dialog.issue.state.value=='accepted', dialog.status.text()
            dialog.show_analysis_readiness()
            idle()
            assert 'готов к использованию' in dialog.history.toPlainText()
            dialog.close()
            assert not workers
            del dialog
            app.processEvents()
    """)
