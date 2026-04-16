(function () {
    const STORAGE_KEY = 'neuralimage_webui_form_v1';
    const PANEL_PREFS_KEY = 'neuralimage_webui_panels_v1';
    const THEME_KEY = 'neuralimage_webui_theme_v1';

    const shell = document.getElementById('app-shell');
    const toggleSettingsBtn = document.getElementById('toggle-settings');
    const statusNode = document.getElementById('run-status');
    const logContainer = document.getElementById('logs-container');
    const maxLogLines = 500;
    const textsNode = document.getElementById('webui-texts-data');
    const endpointsNode = document.getElementById('webui-endpoints-data');

    let uiTexts = {};
    if (textsNode && textsNode.textContent) {
        try {
            uiTexts = JSON.parse(textsNode.textContent);
        } catch (_error) {
            uiTexts = {};
        }
    }
    let endpoints = {};
    if (endpointsNode && endpointsNode.textContent) {
        try {
            endpoints = JSON.parse(endpointsNode.textContent);
        } catch (_error) {
            endpoints = {};
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
    const queuePropertiesBtn = document.getElementById('queue-properties-btn');
    const queueRemoveBtn = document.getElementById('queue-remove-btn');
    const queuePauseBtn = document.getElementById('queue-pause-btn');
    const uiModeButtons = document.querySelectorAll('.mode-btn');
    const themeButtons = document.querySelectorAll('.theme-btn');
    const workflowPresetButtons = document.querySelectorAll('.workflow-preset-btn');
    const simpleWorkflowLabelNode = document.getElementById('simple-workflow-label');
    const sampleCountNode = document.getElementById('sample-count-value');
    const importWorkflowBtn = document.getElementById('import-workflow-btn');
    const workflowImportInput = document.getElementById('workflow-import-input');
    const openHelpBtn = document.getElementById('open-help-btn');
    const openChangelogBtn = document.getElementById('open-changelog-btn');
    const checkUpdatesBtn = document.getElementById('check-updates-btn');
    const releaseMemoryBtn = document.getElementById('release-memory-btn');
    const resetDefaultsBtn = document.getElementById('reset-defaults-btn');
    const qtToolButtons = document.querySelectorAll('.qt-tool-btn');
    const settingsNavButtons = document.querySelectorAll('.settings-nav-btn');
    const panelToggleButtons = document.querySelectorAll('.panel-toggle-btn');
    const menuActionButtons = document.querySelectorAll('[data-menu-action]');
    const metricsPanelNode = document.getElementById('metrics-panel');
    const logsPanelNode = document.getElementById('logs-panel');
    const settingsPanelNode = document.getElementById('settings-pane');
    const previewCardNode = document.getElementById('preview-card');
    const modalShell = document.getElementById('modal-shell');
    const modalBackdrop = document.getElementById('modal-backdrop');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalTitleNode = document.getElementById('modal-title');
    const modalBodyNode = document.getElementById('modal-body');
    const modalActionsNode = document.getElementById('modal-actions');
    const modalCardNode = modalShell ? modalShell.querySelector('.modal-card') : null;

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
    const startForm = document.getElementById('start-form');
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
    const trainPatchXInput = document.querySelector('[name="settings-sample_x"]');
    const trainPatchYInput = document.querySelector('[name="settings-sample_y"]');
    const recognitionPatchXInput = document.querySelector('[name="settings-recognition_sample_x"]');
    const recognitionPatchYInput = document.querySelector('[name="settings-recognition_sample_y"]');
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
    const lossPresetButtons = document.querySelectorAll('.loss-preset-btn');

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
    const recognitionPatchSizeField = document.querySelector('[data-role="recognition-patch-size"]');
    const rarePatchFields = document.querySelector('[data-role="rare-patch-fields"]');
    const rarePatchEnabledInput = document.querySelector('[name="settings-rare_patch_oversampling_enabled"]');
    const schedulerGroups = document.querySelectorAll('[data-scheduler-group]');

    let afterId = 0;
    let notificationAfterId = 0;
    let pendingBroadcastNotifications = [];
    let activeModalOptions = {};
    let selectedQueueTaskId = null;
    let activeUiMode = shell ? (shell.dataset.uiMode || 'simple') : 'simple';
    let activeTheme = localStorage.getItem(THEME_KEY) || 'dark';
    let activeWorkflowPreset = '';
    let sampleCountTimer = null;
    let remoteSourceFiles = [];
    let remoteSourceRelativePaths = [];
    let resultDirectoryHandle = null;
    let panelPrefs = {
        metrics: true,
        logs: true,
        settings: true,
        preview: true,
    };

    function t(key, fallback) {
        const value = uiTexts[key];
        return typeof value === 'string' && value.trim() ? value : fallback;
    }

    function getCsrfToken() {
        const tokenNode = document.querySelector('#start-form input[name="csrfmiddlewaretoken"]');
        return tokenNode ? tokenNode.value : '';
    }

    function endpoint(name, fallback) {
        const value = endpoints[name];
        return typeof value === 'string' && value.trim() ? value : fallback;
    }

    function escapeHtml(value) {
        return String(value == null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderInlineMarkdown(text) {
        return escapeHtml(text)
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    function renderMarkdown(markdown) {
        const normalized = String(markdown || '').replace(/\r\n/g, '\n');
        const lines = normalized.split('\n');
        const blocks = [];
        let listItems = [];
        let codeLines = [];
        let inCode = false;
        let paragraph = [];

        function flushParagraph() {
            if (!paragraph.length) return;
            blocks.push(`<p>${renderInlineMarkdown(paragraph.join(' '))}</p>`);
            paragraph = [];
        }

        function flushList() {
            if (!listItems.length) return;
            blocks.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ul>`);
            listItems = [];
        }

        function flushCode() {
            if (!codeLines.length) return;
            blocks.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
            codeLines = [];
        }

        lines.forEach((line) => {
            if (line.trim().startsWith('```')) {
                flushParagraph();
                flushList();
                if (inCode) {
                    flushCode();
                }
                inCode = !inCode;
                return;
            }
            if (inCode) {
                codeLines.push(line);
                return;
            }
            const headingMatch = /^(#{1,3})\s+(.*)$/.exec(line);
            if (headingMatch) {
                flushParagraph();
                flushList();
                const level = headingMatch[1].length;
                blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
                return;
            }
            const listMatch = /^\s*[-*]\s+(.*)$/.exec(line);
            if (listMatch) {
                flushParagraph();
                listItems.push(listMatch[1]);
                return;
            }
            if (!line.trim()) {
                flushParagraph();
                flushList();
                return;
            }
            paragraph.push(line.trim());
        });

        flushParagraph();
        flushList();
        flushCode();
        return `<div class="modal-markdown">${blocks.join('')}</div>`;
    }

    function openModal(title, bodyHtml, actions, options) {
        if (!modalShell || !modalTitleNode || !modalBodyNode || !modalActionsNode) return;
        activeModalOptions = options || {};
        modalTitleNode.textContent = title || '';
        modalBodyNode.innerHTML = bodyHtml || '';
        modalActionsNode.innerHTML = '';
        if (modalCardNode) {
            modalCardNode.className = `modal-card ${activeModalOptions.cardClass || ''}`.trim();
        }
        (actions || []).forEach((action) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = action.className || 'ghost-btn';
            button.textContent = action.label || '';
            button.addEventListener('click', () => {
                if (typeof action.onClick === 'function') {
                    action.onClick();
                }
            });
            modalActionsNode.appendChild(button);
        });
        modalShell.hidden = false;
    }

    function closeModal() {
        const onClose = activeModalOptions.onClose;
        activeModalOptions = {};
        if (modalShell) modalShell.hidden = true;
        if (modalCardNode) modalCardNode.className = 'modal-card';
        if (typeof onClose === 'function') {
            onClose();
        }
        window.setTimeout(showNextBroadcastNotification, 0);
    }

    function isModalOpen() {
        return !!(modalShell && !modalShell.hidden);
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

    function loadPanelPrefs() {
        const raw = localStorage.getItem(PANEL_PREFS_KEY);
        if (!raw) return;
        try {
            const parsed = JSON.parse(raw);
            panelPrefs = {
                metrics: parsed.metrics !== false,
                logs: parsed.logs !== false,
                settings: parsed.settings !== false,
                preview: parsed.preview !== false,
            };
        } catch (_error) {
            panelPrefs = { metrics: true, logs: true, settings: true, preview: true };
        }
    }

    function savePanelPrefs() {
        localStorage.setItem(PANEL_PREFS_KEY, JSON.stringify(panelPrefs));
    }

    function applyTheme(theme) {
        activeTheme = theme === 'light' ? 'light' : 'dark';
        document.body.classList.toggle('theme-light', activeTheme === 'light');
        themeButtons.forEach((btn) => {
            btn.classList.toggle('is-active', btn.dataset.theme === activeTheme);
        });
        localStorage.setItem(THEME_KEY, activeTheme);
    }

    function setFormControlValue(name, value) {
        const control = document.querySelector(`[name="${name}"]`);
        if (!control) return;
        if (control.type === 'checkbox') {
            control.checked = !!value;
            return;
        }
        if (control.type === 'radio') {
            const radios = document.querySelectorAll(`[name="${name}"]`);
            radios.forEach((radio) => {
                radio.checked = String(radio.value) === String(value);
            });
            return;
        }
        control.value = value == null ? '' : value;
    }

    function applyServerState(state) {
        if (!state) return;
        const mainState = state.main || {};
        const settingsState = state.settings || {};
        Object.entries(mainState).forEach(([key, value]) => {
            setFormControlValue(`main-${key}`, value);
        });
        Object.entries(settingsState).forEach(([key, value]) => {
            setFormControlValue(`settings-${key}`, value);
        });
        if (state.ui_mode === 'simple' || state.ui_mode === 'advanced') {
            activeUiMode = state.ui_mode;
            updateUiModeState();
        }
        syncWorkModeRadiosFromSelect();
        syncCutModeRadiosFromSelect();
        applyModeRules();
        applyDependentRules();
        markActivePreset();
        persistFormState();
        scheduleSampleCountRefresh();
    }

    function setSampleCountText(text) {
        if (sampleCountNode) sampleCountNode.textContent = text;
    }

    function buildCurrentFormPayload() {
        const body = new URLSearchParams();
        getPersistedControls().forEach((el) => {
            if (!el.name || el.name === 'csrfmiddlewaretoken') return;
            if (el.type === 'radio') {
                if (el.checked) body.set(el.name, el.value);
                return;
            }
            if (el.type === 'checkbox') {
                if (el.checked) body.set(el.name, 'on');
                return;
            }
            body.set(el.name, el.value);
        });
        return body;
    }

    async function refreshSampleCount() {
        setSampleCountText(t('samples_count_loading', 'Calculating...'));
        try {
            const payload = await postForm(endpoint('sampleCountUrl', '/api/sample-count/'), buildCurrentFormPayload(), false);
            const count = Number(payload.count || 0);
            const template = t('samples_count_template', 'Dataset frames: {count}');
            setSampleCountText(template.replace('{count}', String(count)));
        } catch (_error) {
            setSampleCountText(t('samples_count', 'Dataset frames: 0'));
        }
    }

    function scheduleSampleCountRefresh() {
        if (sampleCountTimer) window.clearTimeout(sampleCountTimer);
        sampleCountTimer = window.setTimeout(refreshSampleCount, 250);
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

    function syncRecognitionPatchSize() {
        const shouldSync = !!(syncPatchSizesInput && syncPatchSizesInput.checked);
        if (shouldSync) {
            if (recognitionPatchXInput && trainPatchXInput) recognitionPatchXInput.value = trainPatchXInput.value;
            if (recognitionPatchYInput && trainPatchYInput) recognitionPatchYInput.value = trainPatchYInput.value;
        }
        setFieldEnabled(recognitionPatchSizeField, !shouldSync);
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

        syncRecognitionPatchSize();
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
        setFieldEnabled(rarePatchFields, !!(rarePatchEnabledInput && rarePatchEnabledInput.checked));
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

    function isManagedResultTarget(targetInput) {
        return !!(targetInput && targetInput.name === 'main-result_folder');
    }

    function isRemoteSourceTarget(targetInput) {
        return !!(targetInput && targetInput.name === 'main-source_folder');
    }

    function isRecognitionOnlyMode() {
        const mode = workModeSelect ? normalizeWorkModeValue(workModeSelect.value) : '';
        return mode === 'recognition_only';
    }

    function createBrowserPathInput(kind, fileFilter) {
        const input = document.createElement('input');
        input.type = 'file';
        input.hidden = true;
        if (kind === 'folder') {
            input.multiple = true;
            input.setAttribute('webkitdirectory', '');
            input.setAttribute('directory', '');
        }
        if (fileFilter === 'model') {
            input.accept = '.pth,.pt,.ckpt,application/octet-stream';
        }
        document.body.appendChild(input);
        return input;
    }

    function setRemoteSourceSelection(targetInput, files) {
        remoteSourceFiles = Array.from(files || []);
        remoteSourceRelativePaths = remoteSourceFiles.map((file) => file.webkitRelativePath || file.name);
        const count = remoteSourceFiles.length;
        const hasZip = count === 1 && /\.zip$/i.test(remoteSourceFiles[0].name || '');
        targetInput.value = hasZip
            ? `browser-stream://zip/${remoteSourceFiles[0].name}`
            : `browser-stream://${count}-files`;
        targetInput.dispatchEvent(new Event('input', { bubbles: true }));
        targetInput.dispatchEvent(new Event('change', { bubbles: true }));
        persistFormState();
    }

    async function chooseResultFolder(targetInput) {
        resultDirectoryHandle = null;
        if ('showDirectoryPicker' in window) {
            try {
                resultDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
                targetInput.value = `browser-result://${resultDirectoryHandle.name || 'selected-folder'}`;
            } catch (_error) {
                return;
            }
        } else {
            targetInput.value = 'browser-download://recognition-results';
            window.alert(t('result_folder_download_fallback', 'Browser folder writing is unavailable. The result archive will be downloaded.'));
        }
        targetInput.dispatchEvent(new Event('input', { bubbles: true }));
        targetInput.dispatchEvent(new Event('change', { bubbles: true }));
        persistFormState();
    }

    async function uploadPathSelection(button, targetInput, kind, fileFilter, files) {
        const csrf = getCsrfToken();
        if (!csrf) {
            window.alert(t('csrf_missing', 'CSRF token not found. Reload page and try again.'));
            return;
        }

        const body = new FormData();
        body.set('kind', kind);
        body.set('target', targetInput.name || '');
        if (fileFilter) body.set('filter', fileFilter);
        (files || []).forEach((file) => {
            body.append('files', file, file.name);
            body.append('relative_paths', file.webkitRelativePath || file.name);
        });

        button.disabled = true;
        try {
            const response = await fetch(endpoint('pickPathUrl', '/api/pick-path/'), {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrf,
                },
                body,
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

    async function handlePickPath(button) {
        const targetSelector = button.dataset.target || '';
        const kind = button.dataset.kind || 'folder';
        const fileFilter = button.dataset.filter || '';
        const targetInput = document.querySelector(targetSelector);
        if (!targetInput) return;

        if (kind === 'folder' && isRemoteSourceTarget(targetInput)) {
            const input = createBrowserPathInput(kind, fileFilter);
            input.addEventListener('change', () => {
                const files = Array.from(input.files || []);
                if (files.length) {
                    setRemoteSourceSelection(targetInput, files);
                }
                input.remove();
            }, { once: true });
            input.click();
            return;
        }

        if (kind === 'folder' && isManagedResultTarget(targetInput)) {
            await chooseResultFolder(targetInput);
            return;
        }

        const input = createBrowserPathInput(kind, fileFilter);
        input.addEventListener('change', async () => {
            const files = Array.from(input.files || []);
            try {
                if (files.length) {
                    await uploadPathSelection(button, targetInput, kind, fileFilter, files);
                }
            } finally {
                input.remove();
            }
        }, { once: true });
        input.click();
    }

    function addZipUploadButtons() {
        pickPathButtons.forEach((button) => {
            if ((button.dataset.kind || '') !== 'folder') return;
            const targetInput = document.querySelector(button.dataset.target || '');
            if (!targetInput || isManagedResultTarget(targetInput)) return;
            const zipButton = document.createElement('button');
            zipButton.type = 'button';
            zipButton.className = button.className;
            zipButton.textContent = 'ZIP';
            zipButton.title = t('select_zip_tip', 'Upload a .zip archive and unpack it on the server.');
            zipButton.addEventListener('click', () => handleZipPath(button, targetInput));
            button.insertAdjacentElement('afterend', zipButton);
        });
    }

    function handleZipPath(sourceButton, targetInput) {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.zip,application/zip';
        input.hidden = true;
        document.body.appendChild(input);
        input.addEventListener('change', async () => {
            const files = Array.from(input.files || []);
            try {
                if (!files.length) return;
                if (isRemoteSourceTarget(targetInput)) {
                    setRemoteSourceSelection(targetInput, files);
                } else {
                    await uploadPathSelection(sourceButton, targetInput, 'folder', 'zip', files);
                }
            } finally {
                input.remove();
            }
        }, { once: true });
        input.click();
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
        workflowPresetButtons.forEach((btn) => {
            btn.classList.toggle('is-active', btn.dataset.preset === activeWorkflowPreset);
        });
        if (simpleWorkflowLabelNode) {
            const template = t('simple_workflow_selected_template', 'Current profile: {profile}');
            const profileLabel = activeWorkflowPreset
                ? t(`simple_workflow_${activeWorkflowPreset}`, activeWorkflowPreset)
                : t('simple_workflow_none', 'not selected');
            simpleWorkflowLabelNode.textContent = template.replace('{profile}', profileLabel);
        }
    }

    function applyPreset(btn) {
        if (!optimizerInput || !learningRateInput || !weightDecayInput) return;
        optimizerInput.value = btn.dataset.optimizer || optimizerInput.value;
        learningRateInput.value = btn.dataset.lr || learningRateInput.value;
        weightDecayInput.value = btn.dataset.wd || weightDecayInput.value;
        markActivePreset();
        persistFormState();
    }

    function applyLossPreset(btn) {
        if (!lossFunctionInput) return;
        lossFunctionInput.value = btn.dataset.loss || lossFunctionInput.value;
        lossPresetButtons.forEach((item) => {
            item.classList.toggle('is-active', item.dataset.loss === lossFunctionInput.value);
        });
        applyDependentRules();
        persistFormState();
    }

    function updateUiModeState() {
        if (!shell) return;
        shell.dataset.uiMode = activeUiMode;
        shell.classList.toggle('is-simple', activeUiMode === 'simple');
        shell.classList.toggle('is-advanced', activeUiMode === 'advanced');
        uiModeButtons.forEach((btn) => {
            btn.classList.toggle('is-active', btn.dataset.uiMode === activeUiMode);
        });
        applyPanelPrefs();
    }

    function applyPanelPrefs() {
        if (metricsPanelNode) metricsPanelNode.classList.toggle('is-hidden-panel', !panelPrefs.metrics || activeUiMode === 'simple');
        if (logsPanelNode) logsPanelNode.classList.toggle('is-hidden-panel', !panelPrefs.logs || activeUiMode === 'simple');
        if (settingsPanelNode) settingsPanelNode.classList.toggle('is-hidden-panel', !panelPrefs.settings || activeUiMode === 'simple');
        if (previewCardNode) previewCardNode.classList.toggle('is-hidden-panel', !panelPrefs.preview);
        panelToggleButtons.forEach((btn) => {
            const key = btn.dataset.panel || '';
            const active = !!panelPrefs[key] && !(activeUiMode === 'simple' && key !== 'preview');
            btn.classList.toggle('is-active', active);
            if (activeUiMode === 'simple' && key !== 'preview') {
                btn.disabled = true;
            } else {
                btn.disabled = false;
            }
        });
    }

    function updateButtonsByStatus(status, permissions) {
        const isRunning = status === 'running' || status === 'stopping';
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.style.display = 'inline-block';
        }
        if (stopBtn) {
            stopBtn.disabled = !(isRunning && permissions && permissions.can_stop_active_task);
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

    function showNextBroadcastNotification() {
        if (!pendingBroadcastNotifications.length || isModalOpen()) return;
        const item = pendingBroadcastNotifications.shift();
        const author = String(item.created_by || '').trim();
        const message = String(item.message || '').trim();
        const body = `
            <div class="notification-body">
                ${author ? `<div class="notification-author">${escapeHtml(author)}</div>` : ''}
                <div class="notification-message">${escapeHtml(message)}</div>
            </div>
        `;
        openModal(
            t('broadcast_notification_title', 'Уведомление'),
            body,
            [{
                label: t('close_button', 'Закрыть'),
                className: 'ghost-btn',
                onClick: closeModal,
            }],
            {
                blocking: true,
                cardClass: 'notification-modal',
                onClose: () => window.setTimeout(showNextBroadcastNotification, 0),
            },
        );
    }

    function appendNotifications(notifications) {
        if (!Array.isArray(notifications) || notifications.length === 0) return;
        notifications.forEach((item) => {
            if (item && item.message) {
                pendingBroadcastNotifications.push(item);
            }
        });
        showNextBroadcastNotification();
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
            if (queuePropertiesBtn) queuePropertiesBtn.disabled = true;
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
                <div class="queue-item-owner">${item.owner_display_name || item.owner_username || '-'}</div>
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
        const isOwner = !!(selected && selected.is_owner);
        if (queuePropertiesBtn) queuePropertiesBtn.disabled = !selected;
        if (queuePauseBtn) queuePauseBtn.disabled = !selected || !isOwner || selectedStatus === 'running';
        if (queueRemoveBtn) queueRemoveBtn.disabled = !selected || !isOwner || selectedStatus === 'running';
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

    async function postForm(url, body, isFormData) {
        const csrf = getCsrfToken();
        const headers = { 'X-CSRFToken': csrf };
        if (!isFormData) {
            headers['Content-Type'] = 'application/x-www-form-urlencoded;charset=UTF-8';
        }
        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: isFormData ? body : body.toString(),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.error || `Request failed (${response.status})`);
        }
        return payload;
    }

    function filenameFromContentDisposition(headerValue) {
        const header = String(headerValue || '');
        const match = /filename\*?=(?:UTF-8''|")?([^";]+)/i.exec(header);
        if (!match) return 'neuralimage_result.zip';
        try {
            return decodeURIComponent(match[1].replace(/"/g, '').trim()) || 'neuralimage_result.zip';
        } catch (_error) {
            return match[1].replace(/"/g, '').trim() || 'neuralimage_result.zip';
        }
    }

    async function saveResultBlob(blob, filename) {
        if (resultDirectoryHandle && typeof resultDirectoryHandle.getFileHandle === 'function') {
            const fileHandle = await resultDirectoryHandle.getFileHandle(filename, { create: true });
            const writable = await fileHandle.createWritable();
            await writable.write(blob);
            await writable.close();
            return;
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function buildStreamingRecognitionPayload() {
        const body = new FormData();
        getPersistedControls().forEach((el) => {
            if (!el.name || el.name === 'csrfmiddlewaretoken') return;
            if (el.type === 'radio') {
                if (el.checked) body.set(el.name, el.value);
                return;
            }
            if (el.type === 'checkbox') {
                if (el.checked) body.set(el.name, 'on');
                return;
            }
            body.set(el.name, el.value);
        });
        remoteSourceFiles.forEach((file, index) => {
            body.append('source_files', file, file.name);
            body.append('source_relative_paths', remoteSourceRelativePaths[index] || file.name);
        });
        return body;
    }

    async function runStreamingRecognition() {
        if (!remoteSourceFiles.length) {
            window.alert(t('source_stream_missing', 'Reselect source files before streaming recognition.'));
            return;
        }
        if (!isRecognitionOnlyMode()) {
            window.alert(t('source_stream_recognition_only', 'Streaming source upload is available only for recognition-only mode.'));
            return;
        }

        if (startBtn) startBtn.disabled = true;
        setSampleCountText(t('recognition_stream_running', 'Streaming recognition is running...'));
        try {
            const response = await fetch(endpoint('streamingRecognitionUrl', '/api/recognition/stream/'), {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCsrfToken(),
                },
                body: buildStreamingRecognitionPayload(),
            });
            const contentType = response.headers.get('Content-Type') || '';
            if (!response.ok) {
                if (contentType.includes('application/json')) {
                    const payload = await response.json();
                    throw new Error(payload.error || `Streaming recognition failed (${response.status})`);
                }
                throw new Error(`Streaming recognition failed (${response.status})`);
            }
            const blob = await response.blob();
            const filename = filenameFromContentDisposition(response.headers.get('Content-Disposition'));
            await saveResultBlob(blob, filename);
            window.alert(t('recognition_stream_done', 'Recognition result archive is ready.'));
        } catch (error) {
            window.alert(error.message || String(error));
        } finally {
            if (startBtn) startBtn.disabled = false;
            scheduleSampleCountRefresh();
        }
    }

    function ensureAdvancedPanel(panelKey) {
        activeUiMode = 'advanced';
        if (panelKey) {
            panelPrefs[panelKey] = true;
        }
        savePanelPrefs();
        updateUiModeState();
    }

    function scrollToSettingsSection(sectionId) {
        ensureAdvancedPanel('settings');
        const target = document.getElementById(sectionId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    async function setUiModeFromMenu(uiMode) {
        try {
            const body = new URLSearchParams();
            body.set('ui_mode', uiMode);
            await postForm(endpoint('uiModeUrl', '/api/ui-mode/'), body, false);
            activeUiMode = uiMode;
            updateUiModeState();
        } catch (error) {
            window.alert(error.message);
        }
    }

    function togglePanelFromMenu(panelKey) {
        if (!panelKey) return;
        if (panelKey !== 'preview' && activeUiMode === 'simple') {
            activeUiMode = 'advanced';
        }
        panelPrefs[panelKey] = !panelPrefs[panelKey];
        savePanelPrefs();
        applyPanelPrefs();
        const targetByKey = {
            metrics: metricsPanelNode,
            logs: logsPanelNode,
            settings: settingsPanelNode,
            preview: previewCardNode,
        };
        const target = targetByKey[panelKey];
        if (target && !target.classList.contains('is-hidden-panel')) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    async function loadWorkflowPreset(presetKey) {
        const url = `${endpoint('workflowPresetUrl', '/api/workflow/preset/')}?preset=${encodeURIComponent(presetKey)}`;
        const response = await fetch(url, { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.error || `Failed to load preset (${response.status})`);
        }
        activeWorkflowPreset = payload.preset || presetKey;
        applyServerState(payload.state || {});
    }

    async function importWorkflowFile(file) {
        const body = new FormData();
        body.append('workflow_file', file);
        const payload = await postForm(endpoint('workflowImportUrl', '/api/workflow/import/'), body, true);
        activeWorkflowPreset = '';
        applyServerState(payload.state || {});
        window.alert(t('workflow_import_ok', 'Configuration imported.'));
    }

    function keyLabel(key) {
        return String(key || '')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (match) => match.toUpperCase());
    }

    function renderPropertyRows(source) {
        return Object.entries(source || {})
            .map(([key, value]) => {
                const formatted = typeof value === 'object'
                    ? `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`
                    : escapeHtml(String(value == null || value === '' ? '-' : value));
                return `<tr><td>${escapeHtml(keyLabel(key))}</td><td>${formatted}</td></tr>`;
            })
            .join('');
    }

    async function openQueueProperties() {
        if (!selectedQueueTaskId) return;
        const url = `${endpoint('queuePropertiesUrl', '/api/queue/properties/')}?task_id=${encodeURIComponent(String(selectedQueueTaskId))}`;
        const response = await fetch(url, { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            window.alert(payload.error || `Failed to load task properties (${response.status})`);
            return;
        }
        const task = payload.task || {};
        const queueSection = {
            task_id: task.task_id,
            status: task.status,
            owner_display_name: task.owner_display_name,
            owner_username: task.owner_username,
        };
        const mainSection = (task.workflow && task.workflow.main_window_state) || {};
        const settingsSection = (task.workflow && task.workflow.settings_state) || {};
        const body = `
            <div class="properties-grid">
                <section class="properties-section">
                    <h3>${escapeHtml(t('queue_section', 'Queue'))}</h3>
                    <table class="properties-table">${renderPropertyRows(queueSection)}</table>
                </section>
                <section class="properties-section">
                    <h3>${escapeHtml(t('main_section', 'Main window'))}</h3>
                    <table class="properties-table">${renderPropertyRows(mainSection)}</table>
                </section>
                <section class="properties-section">
                    <h3>${escapeHtml(t('settings_section', 'Settings'))}</h3>
                    <table class="properties-table">${renderPropertyRows(settingsSection)}</table>
                </section>
            </div>
        `;
        const actions = [{
            label: t('close_button', 'Close'),
            className: 'ghost-btn',
            onClick: closeModal,
        }];
        if (task.can_restore) {
            actions.unshift({
                label: t('restore_button', 'Restore'),
                className: 'ghost-btn',
                onClick: async () => {
                    try {
                        const body = new URLSearchParams();
                        body.set('task_id', String(selectedQueueTaskId));
                        const restorePayload = await postForm(endpoint('queueRestoreUrl', '/api/queue/restore/'), body, false);
                        activeWorkflowPreset = '';
                        applyServerState(restorePayload.state || {});
                        closeModal();
                        window.alert(t('workflow_restore_ok', 'Task parameters restored to the form.'));
                    } catch (error) {
                        window.alert(error.message);
                    }
                },
            });
        }
        openModal(
            `${t('queue_properties_title', 'Task Properties')} #${task.task_id || selectedQueueTaskId}`,
            body,
            actions,
        );
    }

    async function openMarkdownModal(url, fallbackTitle) {
        const response = await fetch(url, { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            throw new Error(payload.error || `Failed to load content (${response.status})`);
        }
        openModal(payload.title || fallbackTitle, renderMarkdown(payload.content || ''), [{
            label: t('close_button', 'Close'),
            className: 'ghost-btn',
            onClick: closeModal,
        }]);
    }

    async function openUpdateInfo() {
        const response = await fetch(endpoint('updateInfoUrl', '/api/update-info/'), { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            window.alert(payload.error || `Failed to check updates (${response.status})`);
            return;
        }
        const channelOptions = (payload.channels || [])
            .map((channel) => `<option value="${escapeHtml(channel)}"${channel === payload.selected_channel ? ' selected' : ''}>${escapeHtml(channel)}</option>`)
            .join('');
        const message = renderMarkdown(payload.message || t('update_not_configured', 'The update source is not configured.'));
        const history = payload.release_history ? renderMarkdown(payload.release_history) : '';
        const body = `
            <div class="update-channel-row">
                <label for="update-channel-select">Channel</label>
                <select id="update-channel-select" class="control-input">${channelOptions}</select>
            </div>
            ${message}
            ${history}
        `;
        openModal(payload.title || t('menu_check_updates', 'Check for updates'), body, [{
            label: t('close_button', 'Close'),
            className: 'ghost-btn',
            onClick: closeModal,
        }]);
        const channelSelect = document.getElementById('update-channel-select');
        if (channelSelect) {
            channelSelect.addEventListener('change', async () => {
                const selected = channelSelect.value;
                const updateResponse = await fetch(`${endpoint('updateInfoUrl', '/api/update-info/')}?channel=${encodeURIComponent(selected)}`, { cache: 'no-store' });
                const updatePayload = await updateResponse.json();
                if (!updateResponse.ok || !updatePayload.ok) {
                    window.alert(updatePayload.error || `Failed to switch update channel (${updateResponse.status})`);
                    return;
                }
                closeModal();
                openUpdateInfo();
            });
        }
    }

    async function releaseMemory() {
        try {
            await postForm(endpoint('releaseMemoryUrl', '/api/release-memory/'), new URLSearchParams(), false);
            window.alert(t('release_memory_ok', 'GPU memory release requested.'));
        } catch (error) {
            window.alert(error.message);
        }
    }

    async function resetDefaults() {
        try {
            const payload = await postForm(endpoint('resetDefaultsUrl', '/api/reset-defaults/'), new URLSearchParams(), false);
            activeWorkflowPreset = '';
            localStorage.removeItem(STORAGE_KEY);
            applyServerState(payload.state || {});
            window.alert(t('reset_defaults_ok', 'Default parameters restored.'));
        } catch (error) {
            window.alert(error.message);
        }
    }

    async function openToolStatus(toolKey) {
        const url = `${endpoint('toolStatusUrl', '/api/tool-status/')}?tool=${encodeURIComponent(toolKey || '')}`;
        const response = await fetch(url, { cache: 'no-store' });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
            window.alert(payload.error || `Failed to load tool status (${response.status})`);
            return;
        }
        openModal(payload.title || t('webui_tool_unavailable_title', 'Tool status'), renderMarkdown(payload.message || ''), [{
            label: t('close_button', 'Close'),
            className: 'ghost-btn',
            onClick: closeModal,
        }]);
    }

    async function handleMenuAction(action) {
        if (action === 'import-workflow') {
            if (workflowImportInput) workflowImportInput.click();
            return;
        }
        if (action === 'settings-sample') {
            scrollToSettingsSection('settings-section-sample');
            return;
        }
        if (action === 'settings-train') {
            scrollToSettingsSection('settings-section-optimizer');
            return;
        }
        if (action === 'settings-pred') {
            scrollToSettingsSection('settings-section-recognition');
            return;
        }
        if (action === 'toggle-metrics') {
            togglePanelFromMenu('metrics');
            return;
        }
        if (action === 'toggle-logs') {
            togglePanelFromMenu('logs');
            return;
        }
        if (action === 'toggle-settings') {
            togglePanelFromMenu('settings');
            return;
        }
        if (action === 'toggle-preview') {
            togglePanelFromMenu('preview');
            return;
        }
        if (action === 'release-memory') {
            await releaseMemory();
            return;
        }
        if (action === 'theme-dark' || action === 'theme-light') {
            applyTheme(action === 'theme-light' ? 'light' : 'dark');
            return;
        }
        if (action === 'ui-simple' || action === 'ui-advanced') {
            await setUiModeFromMenu(action === 'ui-advanced' ? 'advanced' : 'simple');
            return;
        }
        if (action === 'validation-gradient') {
            await openToolStatus('validation_gradient');
            return;
        }
        if (action === 'augmentation-preview') {
            await openToolStatus('augmentation_preview');
            return;
        }
        if (action === 'rare-patch-editor') {
            await openToolStatus('rare_patch_editor');
            return;
        }
        if (action === 'developer') {
            await openToolStatus('developer');
            return;
        }
        if (action === 'open-help') {
            await openMarkdownModal(endpoint('helpContentUrl', '/api/help/'), 'Help');
            return;
        }
        if (action === 'open-changelog') {
            await openMarkdownModal(endpoint('changelogContentUrl', '/api/changelog/'), 'Changelog');
            return;
        }
        if (action === 'check-updates') {
            await openUpdateInfo();
        }
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
            const response = await fetch(`/api/status/?after=${afterId}&notification_after=${notificationAfterId}`, { cache: 'no-store' });
            if (!response.ok) return;

            const data = await response.json();
            const status = data.status || 'idle';
            if (statusNode) statusNode.textContent = data.status_display || status;
            updateButtonsByStatus(status, data.permissions || {});

            afterId = data.last_event_id || afterId;
            notificationAfterId = data.last_notification_id || notificationAfterId;
            appendLogs(data.events || []);
            appendNotifications(data.notifications || []);
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
    if (queuePropertiesBtn) {
        queuePropertiesBtn.addEventListener('click', openQueueProperties);
    }
    if (modalBackdrop) {
        modalBackdrop.addEventListener('click', () => {
            if (activeModalOptions.blocking) return;
            closeModal();
        });
    }
    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeModal);
    }
    menuActionButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            try {
                await handleMenuAction(btn.dataset.menuAction || '');
            } catch (error) {
                window.alert(error.message || String(error));
            }
        });
    });
    if (importWorkflowBtn && workflowImportInput) {
        importWorkflowBtn.addEventListener('click', () => workflowImportInput.click());
        workflowImportInput.addEventListener('change', async () => {
            const [file] = workflowImportInput.files || [];
            if (!file) return;
            try {
                await importWorkflowFile(file);
            } catch (error) {
                window.alert(error.message);
            } finally {
                workflowImportInput.value = '';
            }
        });
    }
    workflowPresetButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            try {
                await loadWorkflowPreset(btn.dataset.preset || '');
            } catch (error) {
                window.alert(error.message);
            }
        });
    });
    uiModeButtons.forEach((btn) => {
        btn.addEventListener('click', async () => {
            const nextMode = btn.dataset.uiMode || 'simple';
            try {
                const body = new URLSearchParams();
                body.set('ui_mode', nextMode);
                await postForm(endpoint('uiModeUrl', '/api/ui-mode/'), body, false);
                activeUiMode = nextMode;
                updateUiModeState();
            } catch (error) {
                window.alert(error.message);
            }
        });
    });
    if (openHelpBtn) {
        openHelpBtn.addEventListener('click', async () => {
            try {
                await openMarkdownModal(endpoint('helpContentUrl', '/api/help/'), 'Help');
            } catch (error) {
                window.alert(error.message);
            }
        });
    }
    if (openChangelogBtn) {
        openChangelogBtn.addEventListener('click', async () => {
            try {
                await openMarkdownModal(endpoint('changelogContentUrl', '/api/changelog/'), 'Changelog');
            } catch (error) {
                window.alert(error.message);
            }
        });
    }
    if (checkUpdatesBtn) {
        checkUpdatesBtn.addEventListener('click', openUpdateInfo);
    }
    if (releaseMemoryBtn) {
        releaseMemoryBtn.addEventListener('click', releaseMemory);
    }
    if (resetDefaultsBtn) {
        resetDefaultsBtn.addEventListener('click', resetDefaults);
    }
    if (startForm) {
        startForm.addEventListener('submit', async (event) => {
            const sourceInput = document.querySelector('[name="main-source_folder"]');
            const hasBrowserSource = !!(
                remoteSourceFiles.length
                || (sourceInput && String(sourceInput.value || '').startsWith('browser-stream://'))
            );
            if (!hasBrowserSource) return;
            event.preventDefault();
            await runStreamingRecognition();
        });
    }
    qtToolButtons.forEach((btn) => {
        btn.addEventListener('click', () => openToolStatus(btn.dataset.tool || ''));
    });
    themeButtons.forEach((btn) => {
        btn.addEventListener('click', () => applyTheme(btn.dataset.theme || 'dark'));
    });
    settingsNavButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const target = document.querySelector(btn.dataset.target || '');
            if (target && typeof target.scrollIntoView === 'function') {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    panelToggleButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.panel || '';
            if (!key) return;
            panelPrefs[key] = !panelPrefs[key];
            savePanelPrefs();
            applyPanelPrefs();
            if (key === 'preview') {
                setFormControlValue('settings-show_batch_preview', !!panelPrefs.preview);
                persistFormState();
            }
        });
    });

    pickPathButtons.forEach((btn) => {
        btn.addEventListener('click', () => handlePickPath(btn));
    });
    addZipUploadButtons();

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
    lossPresetButtons.forEach((btn) => {
        btn.addEventListener('click', () => applyLossPreset(btn));
    });

    if (optimizerInput) optimizerInput.addEventListener('change', markActivePreset);
    if (learningRateInput) learningRateInput.addEventListener('input', markActivePreset);
    if (weightDecayInput) weightDecayInput.addEventListener('input', markActivePreset);
    if (syncPatchSizesInput) syncPatchSizesInput.addEventListener('change', applyDependentRules);
    if (trainPatchXInput) trainPatchXInput.addEventListener('input', applyDependentRules);
    if (trainPatchYInput) trainPatchYInput.addEventListener('input', applyDependentRules);
    if (rarePatchEnabledInput) rarePatchEnabledInput.addEventListener('change', applyDependentRules);

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

    getPersistedControls().forEach((el) => {
        if (el.name === 'csrfmiddlewaretoken') return;
        el.addEventListener('change', scheduleSampleCountRefresh);
        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
            el.addEventListener('input', scheduleSampleCountRefresh);
        }
    });

    loadPanelPrefs();
    applyTheme(activeTheme);
    restoreFormState();
    const previewInput = document.querySelector('[name="settings-show_batch_preview"]');
    if (previewInput) {
        panelPrefs.preview = !!previewInput.checked;
    }
    syncWorkModeRadiosFromSelect();
    syncCutModeRadiosFromSelect();
    applyModeRules();
    syncCutModeSelectFromRadios();
    applyDependentRules();
    updateUiModeState();
    markActivePreset();
    bindAutoSave();
    applyPanelPrefs();

    updateButtonsByStatus((statusNode && statusNode.textContent) || 'idle', {});
    scheduleSampleCountRefresh();

    setInterval(poll, 1000);
    poll();
})();
