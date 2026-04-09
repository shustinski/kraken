(function () {
    const STORAGE_KEY = 'neuralimage_webui_form_v1';

    const shell = document.getElementById('app-shell');
    const startForm = document.getElementById('start-form');
    const toggleSettingsBtn = document.getElementById('toggle-settings');
    const statusNode = document.getElementById('run-status');
    const logContainer = document.getElementById('logs-container');
    const maxLogLines = 500;
    const textsNode = document.getElementById('webui-texts-data');

    let uiTexts = {};
    if (textsNode && textsNode.textContent) {
        try {
            uiTexts = JSON.parse(textsNode.textContent);
        } catch (_error) {
            uiTexts = {};
        }
    }

    const chartTrainEpoch = document.getElementById('chart-train-epoch');
    const chartValEpoch = document.getElementById('chart-val-epoch');
    const chartValQuality = document.getElementById('chart-val-quality');
    const chartTrainBatch = document.getElementById('chart-train-batch');
    const valIouNode = document.getElementById('val-iou-value');
    const valDiceNode = document.getElementById('val-dice-value');
    const valF1Node = document.getElementById('val-f1-value');
    const perfDataWaitNode = document.getElementById('perf-data-wait');
    const perfAugmentationNode = document.getElementById('perf-augmentation');
    const perfForwardNode = document.getElementById('perf-forward');
    const perfBackwardNode = document.getElementById('perf-backward');
    const perfOptimizerNode = document.getElementById('perf-optimizer');
    const perfTotalNode = document.getElementById('perf-total');

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const stopPanel = document.getElementById('stop-form');
    const pickPathButtons = document.querySelectorAll('.pick-path-btn');

    const workModeSelect = document.querySelector('[name="main-work_mode"]');
    const workModeRadios = document.querySelectorAll('input[name="work_mode_radio"]');

    const cutModeSelect = document.querySelector('[name="settings-sample_cut_mode"]');
    const cutModeRadios = document.querySelectorAll('input[name="sample_cut_mode_radio"]');

    const useValidationInput = document.querySelector('[name="settings-use_validation"]');
    const validationSourceInput = document.querySelector('[name="settings-validation_source"]');
    const cropEnabledInput = document.querySelector('[name="settings-crop_enabled"]');
    const resizeEnabledInput = document.querySelector('[name="settings-resize_enabled"]');
    const additionalAugmentationInput = document.querySelector('[name="settings-additional_augmentation"]');
    const randomCropInput = document.querySelector('[name="settings-random_crop"]');
    const scaleAugmentationInput = document.querySelector('[name="settings-scale_augmentation"]');
    const syncPatchSizesInput = document.querySelector('[name="settings-sync_patch_sizes"]');
    const patchBatchSyncModeInput = document.querySelector('[name="settings-patch_batch_sync_mode"]');
    const cutoutEnabledInput = document.querySelector('[name="settings-cutout_enabled"]');
    const randomArtifactsEnabledInput = document.querySelector('[name="settings-random_artifacts_enabled"]');
    const mixupEnabledInput = document.querySelector('[name="settings-mixup_enabled"]');
    const hardMiningEnabledInput = document.querySelector('[name="settings-hard_mining_enabled"]');
    const hardPixelMiningEnabledInput = document.querySelector('[name="settings-hard_pixel_mining_enabled"]');
    const rarePatchOversamplingInput = document.querySelector('[name="settings-rare_patch_oversampling_enabled"]');
    const lossFunctionInput = document.querySelector('[name="settings-loss_function"]');
    const schedulerInput = document.querySelector('[name="settings-scheduler_name"]');
    const useMultiGpuInput = document.querySelector('[name="settings-use_multi_gpu"]');
    const recognitionBinarizeOutputInput = document.querySelector('[name="settings-recognition_binarize_output"]');
    const recognitionUseAutoThresholdInput = document.querySelector('[name="settings-recognition_use_auto_threshold"]');
    const recognitionPostprocessInput = document.querySelector('[name="settings-recognition_postprocess"]');
    const useContextBranchInput = document.querySelector('[name="settings-use_context_branch"]');

    const optimizerInput = document.querySelector('[name="settings-optimizer_name"]');
    const learningRateInput = document.querySelector('[name="settings-learning_rate"]');
    const weightDecayInput = document.querySelector('[name="settings-weight_decay"]');
    const presetButtons = document.querySelectorAll('.preset-btn');

    const modeFields = {
        source: document.querySelector('[data-role="source"]'),
        result: document.querySelector('[data-role="result"]'),
        sampleGroup: document.querySelector('[data-role="sample-group"]'),
        sample: document.querySelector('[data-role="sample"]'),
        label: document.querySelector('[data-role="label"]'),
        modelPath: document.querySelector('[data-role="model-path"]'),
    };

    const validationSourceField = document.querySelector('[data-role="validation-source"]');
    const validationPercentField = document.querySelector('[data-role="validation-percent"]');
    const validationImageFolderField = document.querySelector('[data-role="validation-image-folder"]');
    const validationLabelFolderField = document.querySelector('[data-role="validation-label-folder"]');
    const edgeCutField = document.querySelector('[data-role="edge-cut"]');
    const targetSizeField = document.querySelector('[data-role="target-size"]');
    const extraAugmentationFields = document.querySelector('[data-role="extra-aug-fields"]');
    const stepField = document.querySelector('[data-role="step-field"]');
    const cropsPerImageField = document.querySelector('[data-role="crops-per-image-field"]');
    const scaleAugmentationStrengthField = document.querySelector('[data-role="scale-augmentation-strength"]');
    const recognitionPatchSizeField = document.querySelector('[data-role="recognition-patch-size"]');
    const patchBatchSyncModeField = document.querySelector('[data-role="patch-batch-sync-mode"]');
    const cutoutFields = document.querySelector('[data-role="cutout-fields"]');
    const randomArtifactsFields = document.querySelector('[data-role="random-artifacts-fields"]');
    const mixupFields = document.querySelector('[data-role="mixup-fields"]');
    const hardMiningField = document.querySelector('[data-role="hard-mining-fields"]');
    const hardPixelMiningField = document.querySelector('[data-role="hard-pixel-mining-fields"]');
    const rarePatchFields = document.querySelector('[data-role="rare-patch-fields"]');
    const recognitionBatchSizeField = document.querySelector('[data-role="recognition-batch-size"]');
    const multiGpuModeField = document.querySelector('[data-role="multi-gpu-mode"]');
    const recognitionThresholdField = document.querySelector('[data-role="recognition-threshold"]');
    const recognitionPostprocessKernelField = document.querySelector('[data-role="recognition-postprocess-kernel"]');
    const contextCropSizeField = document.querySelector('[data-role="context-crop-size"]');
    const contextInputSizeField = document.querySelector('[data-role="context-input-size"]');
    const contextBranchChannelsField = document.querySelector('[data-role="context-branch-channels"]');
    const fusionTypeField = document.querySelector('[data-role="fusion-type"]');
    const diceLossWeightField = document.querySelector('[data-role="dice-loss-weight"]');
    const iouLossWeightField = document.querySelector('[data-role="iou-loss-weight"]');
    const schedulerFields = document.querySelector('[data-role="scheduler-fields"]');
    const schedulerGroups = document.querySelectorAll('[data-scheduler-group]');

    let afterId = 0;

    function t(key, fallback) {
        const value = uiTexts[key];
        return typeof value === 'string' && value.trim() ? value : fallback;
    }

    function getCsrfToken() {
        const tokenNode = document.querySelector('#start-form input[name=\"csrfmiddlewaretoken\"]');
        return tokenNode ? tokenNode.value : '';
    }

    function normalizeWorkModeValue(raw) {
        if (raw === 'recognintion_only') return 'recognition_only';
        if (raw === 'futher_training') return 'further_training';
        return raw;
    }

    function getPersistedControls() {
        return document.querySelectorAll('input[name], select[name], textarea[name]');
    }

    function persistFormState() {
        const data = {};
        getPersistedControls().forEach((el) => {
            if (el.name === 'csrfmiddlewaretoken' || el.name.startsWith('__present__')) return;
            if (el.type === 'radio') {
                if (el.checked) {
                    data[el.name] = el.value;
                }
                return;
            }
            if (el.type === 'checkbox') {
                data[el.name] = el.checked;
                return;
            }
            data[el.name] = el.value;
        });
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }

    function restoreFormState() {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;

        let data;
        try {
            data = JSON.parse(raw);
        } catch (_error) {
            return;
        }

        getPersistedControls().forEach((el) => {
            if (el.name.startsWith('__present__')) return;
            if (!(el.name in data)) return;
            let value = data[el.name];
            if (el.name === 'main-work_mode' || el.name === 'work_mode_radio') {
                value = normalizeWorkModeValue(value);
            }
            if (el.type === 'radio') {
                el.checked = String(el.value) === String(value);
                return;
            }
            if (el.type === 'checkbox') {
                el.checked = !!value;
                return;
            }
            el.value = value;
        });
    }

    function bindAutoSave() {
        getPersistedControls().forEach((el) => {
            if (el.name === 'csrfmiddlewaretoken' || el.name.startsWith('__present__')) return;
            el.addEventListener('change', persistFormState);
            el.addEventListener('input', persistFormState);
        });
    }

    function setFieldEnabled(wrapper, enabled) {
        if (!wrapper) return;
        wrapper.style.opacity = enabled ? '1' : '0.55';
        const controls = wrapper.querySelectorAll('input, select, textarea, button');
        controls.forEach((node) => {
            if (node.tagName === 'BUTTON') {
                node.disabled = !enabled;
                return;
            }
            if (node.matches('input:not([type="checkbox"]):not([type="radio"]), textarea')) {
                node.readOnly = !enabled;
            }
            node.setAttribute('aria-disabled', enabled ? 'false' : 'true');
        });
    }

    function syncCheckboxPresenceMarkers() {
        if (!startForm) return;
        startForm.querySelectorAll('input[data-checkbox-presence="1"]').forEach((node) => node.remove());
        startForm.querySelectorAll('input[type="checkbox"][name]').forEach((checkbox) => {
            const marker = document.createElement('input');
            marker.type = 'hidden';
            marker.name = `__present__${checkbox.name}`;
            marker.value = '1';
            marker.dataset.checkboxPresence = '1';
            startForm.appendChild(marker);
        });
    }

    function setFieldReadonly(wrapper, enabled) {
        if (!wrapper) return;
        wrapper.style.opacity = enabled ? '1' : '0.55';
        const controls = wrapper.querySelectorAll('input');
        controls.forEach((node) => {
            node.readOnly = !enabled;
        });
    }

    function setFieldVisible(wrapper, visible) {
        if (!wrapper) return;
        wrapper.style.display = visible ? '' : 'none';
    }

    function syncWorkModeSelectFromRadios() {
        if (!workModeSelect) return;
        const checked = document.querySelector('input[name="work_mode_radio"]:checked');
        if (checked) workModeSelect.value = checked.value;
    }

    function syncWorkModeRadiosFromSelect() {
        if (!workModeSelect) return;
        workModeRadios.forEach((node) => {
            node.checked = node.value === workModeSelect.value;
        });
    }

    function syncCutModeSelectFromRadios() {
        if (!cutModeSelect) return;
        const checked = document.querySelector('input[name="sample_cut_mode_radio"]:checked');
        if (checked) cutModeSelect.value = checked.value;
    }

    function syncCutModeRadiosFromSelect() {
        if (!cutModeSelect) return;
        cutModeRadios.forEach((node) => {
            node.checked = node.value === cutModeSelect.value;
        });
    }

    function applyModeRules() {
        syncWorkModeSelectFromRadios();
        let mode = workModeSelect ? workModeSelect.value : 'train_and_recognition';
        if (mode === 'recognintion_only') mode = 'recognition_only';
        if (mode === 'futher_training') mode = 'further_training';

        setFieldEnabled(modeFields.source, true);
        setFieldEnabled(modeFields.result, true);
        setFieldEnabled(modeFields.sampleGroup, true);
        setFieldEnabled(modeFields.sample, true);
        setFieldEnabled(modeFields.label, true);
        setFieldEnabled(modeFields.modelPath, true);

        if (mode === 'train_and_recognition') {
            setFieldEnabled(modeFields.modelPath, false);
            return;
        }
        if (mode === 'recognition_only') {
            setFieldEnabled(modeFields.sampleGroup, false);
            setFieldEnabled(modeFields.sample, false);
            setFieldEnabled(modeFields.label, false);
            return;
        }
        if (mode === 'train_only') {
            setFieldEnabled(modeFields.source, false);
            setFieldEnabled(modeFields.result, false);
            setFieldEnabled(modeFields.modelPath, false);
        }
    }

    function applyDependentRules() {
        syncCutModeSelectFromRadios();
        const isOnlineCutMode = !!(cutModeSelect && cutModeSelect.value === 'online');
        const randomCropEnabled = !!(isOnlineCutMode && randomCropInput && randomCropInput.checked);
        const scaleAugmentationEnabled = !!(
            isOnlineCutMode && scaleAugmentationInput && scaleAugmentationInput.checked
        );
        const patchBatchSyncMode = patchBatchSyncModeInput ? patchBatchSyncModeInput.value : 'patch_and_batch';
        const syncPatchSizes = !!(syncPatchSizesInput && syncPatchSizesInput.checked);
        const validationEnabled = !!(useValidationInput && useValidationInput.checked);
        const validationSource = validationSourceInput ? validationSourceInput.value : 'split';
        const useExternalValidation = validationEnabled && validationSource === 'external';
        const schedulerValue = schedulerInput ? schedulerInput.value : 'off';
        const recognitionBinarizeOutput = !!(recognitionBinarizeOutputInput && recognitionBinarizeOutputInput.checked);
        const recognitionUseAutoThreshold = !!(
            recognitionUseAutoThresholdInput && recognitionUseAutoThresholdInput.checked
        );
        const recognitionPostprocess = !!(recognitionPostprocessInput && recognitionPostprocessInput.checked);
        const useContextBranch = !!(useContextBranchInput && useContextBranchInput.checked);

        setFieldEnabled(validationSourceField, validationEnabled);
        setFieldEnabled(validationPercentField, validationEnabled && !useExternalValidation);
        setFieldEnabled(validationImageFolderField, useExternalValidation);
        setFieldEnabled(validationLabelFolderField, useExternalValidation);
        setFieldEnabled(edgeCutField, !!(cropEnabledInput && cropEnabledInput.checked));
        setFieldEnabled(targetSizeField, !!(resizeEnabledInput && resizeEnabledInput.checked));
        setFieldEnabled(extraAugmentationFields, !!(additionalAugmentationInput && additionalAugmentationInput.checked));
        setFieldReadonly(stepField, !randomCropEnabled);
        setFieldReadonly(cropsPerImageField, randomCropEnabled);
        setFieldEnabled(scaleAugmentationStrengthField, scaleAugmentationEnabled);
        setFieldEnabled(patchBatchSyncModeField, true);
        setFieldEnabled(
            recognitionPatchSizeField,
            !(syncPatchSizes || ['patch', 'patch_and_batch'].includes(patchBatchSyncMode)),
        );
        setFieldEnabled(
            recognitionBatchSizeField,
            !['batch', 'patch_and_batch'].includes(patchBatchSyncMode),
        );
        setFieldEnabled(cutoutFields, !!(cutoutEnabledInput && cutoutEnabledInput.checked));
        setFieldEnabled(randomArtifactsFields, !!(randomArtifactsEnabledInput && randomArtifactsEnabledInput.checked));
        setFieldEnabled(mixupFields, !!(mixupEnabledInput && mixupEnabledInput.checked));
        setFieldEnabled(hardMiningField, !!(hardMiningEnabledInput && hardMiningEnabledInput.checked));
        setFieldEnabled(hardPixelMiningField, !!(hardPixelMiningEnabledInput && hardPixelMiningEnabledInput.checked));
        setFieldEnabled(rarePatchFields, !!(rarePatchOversamplingInput && rarePatchOversamplingInput.checked));
        setFieldEnabled(multiGpuModeField, !!(useMultiGpuInput && useMultiGpuInput.checked));
        setFieldEnabled(recognitionThresholdField, recognitionBinarizeOutput && !recognitionUseAutoThreshold);
        setFieldEnabled(
            recognitionPostprocessKernelField,
            recognitionBinarizeOutput && recognitionPostprocess,
        );
        setFieldEnabled(contextCropSizeField, useContextBranch);
        setFieldEnabled(contextInputSizeField, useContextBranch);
        setFieldEnabled(contextBranchChannelsField, useContextBranch);
        setFieldEnabled(fusionTypeField, useContextBranch);
        setFieldEnabled(
            diceLossWeightField,
            !!(lossFunctionInput && ['bce_dice', 'focal_dice', 'ce_dice'].includes(lossFunctionInput.value)),
        );
        setFieldEnabled(
            iouLossWeightField,
            !!(lossFunctionInput && ['bce_iou', 'focal_iou'].includes(lossFunctionInput.value)),
        );
        setFieldEnabled(schedulerFields, schedulerValue !== 'off');
        schedulerGroups.forEach((node) => {
            const group = node.getAttribute('data-scheduler-group') || '';
            const visible = schedulerValue !== 'off' && group === schedulerValue;
            setFieldVisible(node, visible);
            setFieldEnabled(node, visible);
        });
    }

    async function handlePickPath(button) {
        const targetSelector = button.dataset.target || '';
        const kind = button.dataset.kind || 'folder';
        const fileFilter = button.dataset.filter || '';
        const targetInput = document.querySelector(targetSelector);
        if (!targetInput) return;

        const csrf = getCsrfToken();
        if (!csrf) {
            window.alert(t('csrf_missing', 'CSRF token not found. Reload page and try again.'));
            return;
        }

        const body = new URLSearchParams();
        body.set('kind', kind);
        if (fileFilter) body.set('filter', fileFilter);

        button.disabled = true;
        try {
            const response = await fetch('/api/pick-path/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                    'X-CSRFToken': csrf,
                },
                body: body.toString(),
            });
            const payload = await response.json();
            if (!response.ok || !payload.ok) {
                if (!payload.cancelled) {
                    const message = payload.error || `Path picker failed (${response.status})`;
                    window.alert(message);
                }
                return;
            }
            targetInput.value = payload.path || '';
            targetInput.dispatchEvent(new Event('input', { bubbles: true }));
            targetInput.dispatchEvent(new Event('change', { bubbles: true }));
            persistFormState();
        } catch (_error) {
            window.alert(t('pick_path_failed', 'Failed to pick path. Check WebUI backend logs.'));
        } finally {
            button.disabled = false;
        }
    }

    function markActivePreset() {
        const optimizer = (optimizerInput && optimizerInput.value) || '';
        const lr = Number((learningRateInput && learningRateInput.value) || 0);
        const wd = Number((weightDecayInput && weightDecayInput.value) || 0);

        presetButtons.forEach((btn) => {
            const bOpt = btn.dataset.optimizer;
            const bLr = Number(btn.dataset.lr);
            const bWd = Number(btn.dataset.wd);
            const active = optimizer === bOpt && Math.abs(lr - bLr) < 1e-12 && Math.abs(wd - bWd) < 1e-12;
            btn.classList.toggle('is-active', active);
        });
    }

    function applyPreset(btn) {
        if (!optimizerInput || !learningRateInput || !weightDecayInput) return;
        optimizerInput.value = btn.dataset.optimizer || optimizerInput.value;
        learningRateInput.value = btn.dataset.lr || learningRateInput.value;
        weightDecayInput.value = btn.dataset.wd || weightDecayInput.value;
        markActivePreset();
        persistFormState();
    }

    function updateButtonsByStatus(status) {
        const isRunning = status === 'running' || status === 'stopping';
        if (startBtn) {
            startBtn.disabled = isRunning;
            startBtn.style.display = isRunning ? 'none' : 'inline-block';
        }
        if (stopBtn) {
            stopBtn.disabled = !isRunning;
        }
        if (stopPanel) {
            stopPanel.classList.toggle('visible', isRunning);
        }
    }

    function linePath(points, width, height, color, minX, maxX, minY, maxY) {
        if (!points || points.length === 0 || !Number.isFinite(minX) || !Number.isFinite(maxX)) {
            return '';
        }

        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;

        const pad = 16;
        const w = width - pad * 2;
        const h = height - pad * 2;

        let d = '';
        for (let i = 0; i < points.length; i += 1) {
            const px = pad + ((points[i].x - minX) / rangeX) * w;
            const py = height - pad - ((points[i].y - minY) / rangeY) * h;
            d += (i === 0 ? 'M' : ' L') + px.toFixed(2) + ' ' + py.toFixed(2);
        }

        return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2"/>`;
    }

    function drawChart(svg, points, color) {
        drawMultiChart(svg, [{ points, color }]);
    }

    function drawMultiChart(svg, series) {
        if (!svg) return;
        const width = 600;
        const height = 220;
        const allPoints = [];
        series.forEach((entry) => {
            (entry.points || []).forEach((point) => allPoints.push(point));
        });
        const xs = allPoints.map((p) => Number(p.x)).filter((value) => Number.isFinite(value));
        const ys = allPoints.map((p) => Number(p.y)).filter((value) => Number.isFinite(value));
        const minX = xs.length > 0 ? Math.min(...xs) : 0;
        const maxX = xs.length > 0 ? Math.max(...xs) : 1;
        const minY = ys.length > 0 ? Math.min(...ys) : 0;
        const maxY = ys.length > 0 ? Math.max(...ys) : 1;
        const grid = `
            <rect x="0" y="0" width="${width}" height="${height}" fill="#0f151e"></rect>
            <line x1="0" y1="${height - 16}" x2="${width}" y2="${height - 16}" stroke="#2f3947"/>
            <line x1="16" y1="0" x2="16" y2="${height}" stroke="#2f3947"/>
        `;
        const lines = series
            .map((entry) => linePath(entry.points || [], width, height, entry.color, minX, maxX, minY, maxY))
            .join('');
        const legend = series
            .filter((entry) => entry.label)
            .map(
                (entry, index) => `
                    <circle cx="${28 + index * 90}" cy="18" r="4" fill="${entry.color}"></circle>
                    <text x="${38 + index * 90}" y="22" fill="#dbe4f0" font-size="12">${entry.label}</text>
                `
            )
            .join('');
        svg.innerHTML = grid + lines + legend;
    }

    function appendLogs(events) {
        events.forEach((event) => {
            const div = document.createElement('div');
            div.className = `log-line ${event.topic}`;
            div.textContent = `[${event.id}] ${event.message}`;
            logContainer.appendChild(div);
        });
        while (logContainer && logContainer.children.length > maxLogLines) {
            logContainer.removeChild(logContainer.firstElementChild);
        }
        if (events.length > 0) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }

    function updateMetrics(metrics) {
        const trainEpoch = (metrics.train_epoch || []).map((p) => ({ x: p.epoch, y: p.loss }));
        const rawValEpoch = metrics.val_epoch || [];
        const valEpoch = rawValEpoch.map((p) => ({ x: p.epoch, y: p.loss }));
        const valIou = rawValEpoch
            .filter((p) => Number.isFinite(Number(p.iou)))
            .map((p) => ({ x: p.epoch, y: Number(p.iou) }));
        const valDice = rawValEpoch
            .filter((p) => Number.isFinite(Number(p.dice)))
            .map((p) => ({ x: p.epoch, y: Number(p.dice) }));
        const trainBatch = (metrics.train_batch || []).map((p) => ({ x: p.batch_index, y: p.loss }));
        const quality = metrics.validation_quality || {};
        const lastValPoint = rawValEpoch.length > 0 ? rawValEpoch[rawValEpoch.length - 1] : null;

        const iou = Number(quality.iou ?? (lastValPoint ? lastValPoint.iou : NaN));
        const dice = Number(quality.dice ?? (lastValPoint ? lastValPoint.dice : NaN));
        const f1 = Number(quality.f1 ?? (lastValPoint ? lastValPoint.f1 : NaN));
        const perf = metrics.train_perf || {};

        const formatPercent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-');
        const formatMs = (value) => (Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} ms` : '-');
        if (valIouNode) valIouNode.textContent = formatPercent(iou);
        if (valDiceNode) valDiceNode.textContent = formatPercent(dice);
        if (valF1Node) valF1Node.textContent = formatPercent(f1);
        if (perfDataWaitNode) perfDataWaitNode.textContent = formatMs(perf.data_wait_ms);
        if (perfAugmentationNode) perfAugmentationNode.textContent = formatMs(perf.augmentation_ms);
        if (perfForwardNode) perfForwardNode.textContent = formatMs(perf.forward_ms);
        if (perfBackwardNode) perfBackwardNode.textContent = formatMs(perf.backward_ms);
        if (perfOptimizerNode) perfOptimizerNode.textContent = formatMs(perf.optimizer_ms);
        if (perfTotalNode) perfTotalNode.textContent = formatMs(perf.total_ms);

        drawChart(chartTrainEpoch, trainEpoch, '#58a6ff');
        drawChart(chartValEpoch, valEpoch, '#ffb86b');
        drawMultiChart(chartValQuality, [
            { points: valIou, color: '#58a6ff', label: 'IoU' },
            { points: valDice, color: '#7ee787', label: 'Dice' },
        ]);
        drawChart(chartTrainBatch, trainBatch, '#7ee787');
    }

    async function poll() {
        try {
            const response = await fetch(`/api/status/?after=${afterId}`, { cache: 'no-store' });
            if (!response.ok) return;

            const data = await response.json();
            const status = data.status || 'idle';
            if (statusNode) statusNode.textContent = data.status_display || status;
            updateButtonsByStatus(status);

            afterId = data.last_event_id || afterId;
            appendLogs(data.events || []);
            updateMetrics(data.metrics || {});
        } catch (_error) {
            // Ignore transient polling errors.
        }
    }

    if (toggleSettingsBtn) {
        toggleSettingsBtn.addEventListener('click', () => {
            shell.classList.toggle('show-settings');
        });
    }

    pickPathButtons.forEach((btn) => {
        btn.addEventListener('click', () => handlePickPath(btn));
    });

    workModeRadios.forEach((node) => node.addEventListener('change', () => {
        applyModeRules();
        persistFormState();
    }));
    cutModeRadios.forEach((node) => node.addEventListener('change', () => {
        syncCutModeSelectFromRadios();
        persistFormState();
    }));

    presetButtons.forEach((btn) => {
        btn.addEventListener('click', () => applyPreset(btn));
    });

    if (optimizerInput) optimizerInput.addEventListener('change', markActivePreset);
    if (learningRateInput) learningRateInput.addEventListener('input', markActivePreset);
    if (weightDecayInput) weightDecayInput.addEventListener('input', markActivePreset);

    if (useValidationInput) useValidationInput.addEventListener('change', applyDependentRules);
    if (validationSourceInput) validationSourceInput.addEventListener('change', applyDependentRules);
    if (cropEnabledInput) cropEnabledInput.addEventListener('change', applyDependentRules);
    if (resizeEnabledInput) resizeEnabledInput.addEventListener('change', applyDependentRules);
    if (additionalAugmentationInput) additionalAugmentationInput.addEventListener('change', applyDependentRules);
    if (randomCropInput) randomCropInput.addEventListener('change', applyDependentRules);
    if (scaleAugmentationInput) scaleAugmentationInput.addEventListener('change', applyDependentRules);
    if (cutoutEnabledInput) cutoutEnabledInput.addEventListener('change', applyDependentRules);
    if (randomArtifactsEnabledInput) randomArtifactsEnabledInput.addEventListener('change', applyDependentRules);
    if (mixupEnabledInput) mixupEnabledInput.addEventListener('change', applyDependentRules);
    if (hardMiningEnabledInput) hardMiningEnabledInput.addEventListener('change', applyDependentRules);
    if (hardPixelMiningEnabledInput) hardPixelMiningEnabledInput.addEventListener('change', applyDependentRules);
    if (rarePatchOversamplingInput) rarePatchOversamplingInput.addEventListener('change', applyDependentRules);
    if (syncPatchSizesInput) syncPatchSizesInput.addEventListener('change', applyDependentRules);
    if (patchBatchSyncModeInput) patchBatchSyncModeInput.addEventListener('change', applyDependentRules);
    if (lossFunctionInput) lossFunctionInput.addEventListener('change', applyDependentRules);
    if (schedulerInput) schedulerInput.addEventListener('change', applyDependentRules);
    if (useMultiGpuInput) useMultiGpuInput.addEventListener('change', applyDependentRules);
    if (recognitionBinarizeOutputInput) recognitionBinarizeOutputInput.addEventListener('change', applyDependentRules);
    if (recognitionUseAutoThresholdInput) recognitionUseAutoThresholdInput.addEventListener('change', applyDependentRules);
    if (recognitionPostprocessInput) recognitionPostprocessInput.addEventListener('change', applyDependentRules);
    if (useContextBranchInput) useContextBranchInput.addEventListener('change', applyDependentRules);
    if (startForm) startForm.addEventListener('submit', syncCheckboxPresenceMarkers);

    restoreFormState();
    syncWorkModeRadiosFromSelect();
    syncCutModeRadiosFromSelect();
    applyModeRules();
    syncCutModeSelectFromRadios();
    applyDependentRules();
    markActivePreset();
    bindAutoSave();

    updateButtonsByStatus((statusNode && statusNode.textContent) || 'idle');

    setInterval(poll, 1000);
    poll();
})();

