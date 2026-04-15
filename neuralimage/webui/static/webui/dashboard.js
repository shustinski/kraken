(function () {
    const STORAGE_KEY = 'neuralimage_webui_form_v1';

    const shell = document.getElementById('app-shell');
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

    const queueListNode = document.getElementById('queue-list');
    const queueRemoveBtn = document.getElementById('queue-remove-btn');
    const queuePauseBtn = document.getElementById('queue-pause-btn');

    const progressEpochTextNode = document.getElementById('progress-epoch-text');
    const progressBatchTextNode = document.getElementById('progress-batch-text');
    const progressRecognitionTextNode = document.getElementById('progress-recognition-text');
    const progressEpochFillNode = document.getElementById('progress-epoch-fill');
    const progressBatchFillNode = document.getElementById('progress-batch-fill');
    const progressRecognitionFillNode = document.getElementById('progress-recognition-fill');

    const recognitionSpeedNode = document.getElementById('recognition-speed-value');
    const memoryUsageNode = document.getElementById('memory-usage-value');
    const validationQualitySummaryNode = document.getElementById('validation-quality-summary');
    const performanceSummaryNode = document.getElementById('performance-summary');

    const previewFrameNameNode = document.getElementById('preview-frame-name');
    const previewImageNode = document.getElementById('preview-image');
    const previewLabelNode = document.getElementById('preview-label');
    const previewOutputNode = document.getElementById('preview-output');
    const previewImageEmptyNode = document.getElementById('preview-image-empty');
    const previewLabelEmptyNode = document.getElementById('preview-label-empty');
    const previewOutputEmptyNode = document.getElementById('preview-output-empty');
    const previewLabelColumnNode = document.getElementById('preview-label-column');

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
    const cutoutEnabledInput = document.querySelector('[name="settings-cutout_enabled"]');
    const randomArtifactsEnabledInput = document.querySelector('[name="settings-random_artifacts_enabled"]');
    const mixupEnabledInput = document.querySelector('[name="settings-mixup_enabled"]');
    const hardMiningEnabledInput = document.querySelector('[name="settings-hard_mining_enabled"]');
    const hardPixelMiningEnabledInput = document.querySelector('[name="settings-hard_pixel_mining_enabled"]');
    const lossFunctionInput = document.querySelector('[name="settings-loss_function"]');
    const schedulerInput = document.querySelector('[name="settings-scheduler_name"]');

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
    const cutoutFields = document.querySelector('[data-role="cutout-fields"]');
    const randomArtifactsFields = document.querySelector('[data-role="random-artifacts-fields"]');
    const mixupFields = document.querySelector('[data-role="mixup-fields"]');
    const hardMiningField = document.querySelector('[data-role="hard-mining-fields"]');
    const hardPixelMiningField = document.querySelector('[data-role="hard-pixel-mining-fields"]');
    const diceLossWeightField = document.querySelector('[data-role="dice-loss-weight"]');
    const iouLossWeightField = document.querySelector('[data-role="iou-loss-weight"]');
    const schedulerFields = document.querySelector('[data-role="scheduler-fields"]');
    const schedulerGroups = document.querySelectorAll('[data-scheduler-group]');

    let afterId = 0;
    let selectedQueueTaskId = null;

    function t(key, fallback) {
        const value = uiTexts[key];
        return typeof value === 'string' && value.trim() ? value : fallback;
    }

    function getCsrfToken() {
        const tokenNode = document.querySelector('#start-form input[name="csrfmiddlewaretoken"]');
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
            if (el.name === 'csrfmiddlewaretoken') return;
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
            if (el.name === 'csrfmiddlewaretoken') return;
            el.addEventListener('change', persistFormState);
            el.addEventListener('input', persistFormState);
        });
    }

    function setFieldEnabled(wrapper, enabled) {
        if (!wrapper) return;
        wrapper.style.opacity = enabled ? '1' : '0.55';
        const controls = wrapper.querySelectorAll('input, select, textarea, button');
        controls.forEach((node) => {
            node.disabled = !enabled;
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
        const scaleAugmentationEnabled = !!(isOnlineCutMode && scaleAugmentationInput && scaleAugmentationInput.checked);
        const validationEnabled = !!(useValidationInput && useValidationInput.checked);
        const validationSource = validationSourceInput ? validationSourceInput.value : 'split';
        const useExternalValidation = validationEnabled && validationSource === 'external';
        const schedulerValue = schedulerInput ? schedulerInput.value : 'off';

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
        setFieldEnabled(cutoutFields, !!(cutoutEnabledInput && cutoutEnabledInput.checked));
        setFieldEnabled(randomArtifactsFields, !!(randomArtifactsEnabledInput && randomArtifactsEnabledInput.checked));
        setFieldEnabled(mixupFields, !!(mixupEnabledInput && mixupEnabledInput.checked));
        setFieldEnabled(hardMiningField, !!(hardMiningEnabledInput && hardMiningEnabledInput.checked));
        setFieldEnabled(hardPixelMiningField, !!(hardPixelMiningEnabledInput && hardPixelMiningEnabledInput.checked));
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
            startBtn.disabled = false;
            startBtn.style.display = 'inline-block';
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

    function setProgress(fillNode, textNode, progress) {
        const percent = Number(progress && progress.percent ? progress.percent : 0);
        const text = progress && progress.text ? progress.text : '0%';
        if (fillNode) fillNode.style.width = `${percent}%`;
        if (textNode) textNode.textContent = text;
    }

    function setPreviewImage(imageNode, emptyNode, sourceUrl) {
        if (!imageNode || !emptyNode) return;
        if (sourceUrl) {
            imageNode.src = sourceUrl;
            imageNode.hidden = false;
            emptyNode.hidden = true;
            return;
        }
        imageNode.removeAttribute('src');
        imageNode.hidden = true;
        emptyNode.hidden = false;
    }

    function updatePreview(metrics) {
        const preview = metrics.preview || {};
        const mode = preview.mode === 'recognition' ? 'recognition' : 'train';
        const sampleName = (preview.sample_name || '').trim();
        const frameTemplate = t('preview_current_frame', 'Frame: {name}');
        const frameDefault = t('preview_current_frame_default', 'Frame: -');
        if (previewFrameNameNode) {
            previewFrameNameNode.textContent = sampleName ? frameTemplate.replace('{name}', sampleName) : frameDefault;
        }
        if (previewLabelColumnNode) {
            previewLabelColumnNode.style.display = mode === 'recognition' ? 'none' : '';
        }
        setPreviewImage(previewImageNode, previewImageEmptyNode, preview.image_url || null);
        setPreviewImage(previewLabelNode, previewLabelEmptyNode, mode === 'recognition' ? null : (preview.label_url || null));
        setPreviewImage(previewOutputNode, previewOutputEmptyNode, preview.output_url || null);
    }

    function updateRuntime(metrics) {
        const progress = metrics.progress || {};
        const memory = metrics.system_memory || {};
        const quality = metrics.validation_quality || {};
        const perf = metrics.train_perf || {};
        const memoryUnit = t('memory_unit', 'MB');
        const speedUnit = t('speed_unit', 'batch/s');
        const ramLabel = t('runtime_ram_label', 'RAM');
        const vramLabel = t('runtime_vram_label', 'VRAM');
        const speedLabel = t('runtime_speed_label', 'Speed');
        const recognitionLabel = t('recognition_speed_label', 'Recognition speed');
        const recognitionUnit = t('recognition_speed_unit', 'img/s');
        const validationDefault = t('validation_quality_default', 'Validation quality: -');
        const performanceDefault = t('performance_label_default', 'Performance: -');
        const formatPercent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '-');

        setProgress(progressEpochFillNode, progressEpochTextNode, progress.epoch || {});
        setProgress(progressBatchFillNode, progressBatchTextNode, progress.batch || {});
        setProgress(progressRecognitionFillNode, progressRecognitionTextNode, progress.recognition || {});

        if (recognitionSpeedNode) {
            const speed = Number(metrics.recognition_speed_images_per_sec);
            recognitionSpeedNode.textContent = Number.isFinite(speed)
                ? `${recognitionLabel}: ${speed.toFixed(2)} ${recognitionUnit}`
                : t('recognition_speed_default', 'Recognition speed: -');
        }

        if (memoryUsageNode) {
            const ramText = Number.isFinite(Number(memory.ram_mb))
                ? `${ramLabel}: ${Number(memory.ram_mb).toFixed(0)} ${memoryUnit}`
                : `${ramLabel}: -`;
            let vramText = `${vramLabel}: -`;
            if (Number.isFinite(Number(memory.vram_allocated_mb))) {
                const reserved = Number.isFinite(Number(memory.vram_reserved_mb))
                    ? `/${Number(memory.vram_reserved_mb).toFixed(0)}`
                    : '';
                vramText = `${vramLabel}: ${Number(memory.vram_allocated_mb).toFixed(0)}${reserved} ${memoryUnit}`;
            }
            const trainSpeed = Number(metrics.train_speed_batches_per_sec);
            const speedText = Number.isFinite(trainSpeed)
                ? `${speedLabel}: ${trainSpeed.toFixed(2)} ${speedUnit}`
                : `${speedLabel}: - ${speedUnit}`;
            const hasRuntime = Number.isFinite(Number(memory.ram_mb))
                || Number.isFinite(Number(memory.vram_allocated_mb))
                || Number.isFinite(Number(memory.vram_reserved_mb))
                || Number.isFinite(trainSpeed);
            memoryUsageNode.textContent = hasRuntime ? `${ramText} | ${vramText} | ${speedText}` : t('memory_label_default', 'Memory: -');
        }

        if (validationQualitySummaryNode) {
            const hasQuality = Number.isFinite(Number(quality.iou))
                || Number.isFinite(Number(quality.dice))
                || Number.isFinite(Number(quality.f1));
            validationQualitySummaryNode.textContent = hasQuality
                ? `IoU: ${formatPercent(Number(quality.iou))} | Dice: ${formatPercent(Number(quality.dice))} | F1: ${formatPercent(Number(quality.f1))}`
                : validationDefault;
        }

        if (performanceSummaryNode) {
            const hasPerf = Number.isFinite(Number(perf.total_ms)) && Number(perf.total_ms) > 0;
            performanceSummaryNode.textContent = hasPerf
                ? `Batch timing | data: ${Number(perf.data_wait_ms || 0).toFixed(1)} ms | augmentation: ${Number(perf.augmentation_ms || 0).toFixed(1)} ms | forward: ${Number(perf.forward_ms || 0).toFixed(1)} ms | backward: ${Number(perf.backward_ms || 0).toFixed(1)} ms | optimizer: ${Number(perf.optimizer_ms || 0).toFixed(1)} ms | total: ${Number(perf.total_ms || 0).toFixed(1)} ms`
                : performanceDefault;
        }

        updatePreview(metrics);
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
        updateRuntime(metrics);
    }

    function queueStatusLabel(status) {
        const values = uiTexts.queue_status_values || {};
        return values[status] || status;
    }

    function workModeLabel(mode) {
        const values = uiTexts.work_mode_labels || {};
        return values[mode] || mode;
    }

    function renderQueue(queueItems) {
        if (!queueListNode) return;
        queueListNode.innerHTML = '';
        if (!Array.isArray(queueItems) || queueItems.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'queue-item';
            empty.innerHTML = `<div class="queue-item-meta">${t('queue_empty', 'No tasks in queue.')}</div>`;
            queueListNode.appendChild(empty);
            selectedQueueTaskId = null;
            if (queuePauseBtn) queuePauseBtn.disabled = true;
            if (queueRemoveBtn) queueRemoveBtn.disabled = true;
            return;
        }

        const queueTaskIds = queueItems.map((item) => Number(item.task_id));
        if (!queueTaskIds.includes(Number(selectedQueueTaskId))) {
            const running = queueItems.find((item) => item.status === 'running');
            selectedQueueTaskId = running ? Number(running.task_id) : Number(queueItems[0].task_id);
        }

        queueItems.forEach((item) => {
            const node = document.createElement('button');
            node.type = 'button';
            node.className = `queue-item ${item.status || 'queued'}`;
            if (Number(item.task_id) === Number(selectedQueueTaskId)) {
                node.classList.add('is-selected');
            }
            node.innerHTML = `
                <div class="queue-item-title">#${item.task_id} | ${workModeLabel(item.work_mode)}</div>
                <div class="queue-item-meta">${queueStatusLabel(item.status)}</div>
            `;
            node.addEventListener('click', () => {
                selectedQueueTaskId = Number(item.task_id);
                renderQueue(queueItems);
            });
            queueListNode.appendChild(node);
        });

        const selected = queueItems.find((item) => Number(item.task_id) === Number(selectedQueueTaskId)) || null;
        const selectedStatus = selected ? selected.status : '';
        if (queuePauseBtn) queuePauseBtn.disabled = !selected || selectedStatus === 'running';
        if (queueRemoveBtn) queueRemoveBtn.disabled = !selected || selectedStatus === 'running';
    }

    async function postQueueAction(url, taskId) {
        const csrf = getCsrfToken();
        if (!csrf) {
            window.alert(t('csrf_missing', 'CSRF token not found. Reload page and try again.'));
            return false;
        }
        const body = new URLSearchParams();
        body.set('task_id', String(taskId));
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'X-CSRFToken': csrf,
            },
            body: body.toString(),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            window.alert(payload.error || `Queue action failed (${response.status})`);
            return false;
        }
        return true;
    }

    async function removeSelectedQueueTask() {
        if (!selectedQueueTaskId) return;
        if (queueRemoveBtn) queueRemoveBtn.disabled = true;
        try {
            const ok = await postQueueAction('/api/queue/remove/', selectedQueueTaskId);
            if (ok) {
                await poll();
            }
        } finally {
            if (queueRemoveBtn) queueRemoveBtn.disabled = false;
        }
    }

    async function pauseSelectedQueueTask() {
        if (!selectedQueueTaskId) return;
        if (queuePauseBtn) queuePauseBtn.disabled = true;
        try {
            const ok = await postQueueAction('/api/queue/pause-toggle/', selectedQueueTaskId);
            if (ok) {
                await poll();
            }
        } finally {
            if (queuePauseBtn) queuePauseBtn.disabled = false;
        }
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
            renderQueue(data.queue || []);
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

    if (queueRemoveBtn) {
        queueRemoveBtn.addEventListener('click', removeSelectedQueueTask);
    }
    if (queuePauseBtn) {
        queuePauseBtn.addEventListener('click', pauseSelectedQueueTask);
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
    if (lossFunctionInput) lossFunctionInput.addEventListener('change', applyDependentRules);
    if (schedulerInput) schedulerInput.addEventListener('change', applyDependentRules);

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
