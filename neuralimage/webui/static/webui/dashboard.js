(function () {
    const STORAGE_KEY = 'neuralimage_webui_form_v1';

    const shell = document.getElementById('app-shell');
    const toggleSettingsBtn = document.getElementById('toggle-settings');
    const statusNode = document.getElementById('run-status');
    const logContainer = document.getElementById('logs-container');

    const chartTrainEpoch = document.getElementById('chart-train-epoch');
    const chartValEpoch = document.getElementById('chart-val-epoch');
    const chartTrainBatch = document.getElementById('chart-train-batch');
    const valIouNode = document.getElementById('val-iou-value');
    const valDiceNode = document.getElementById('val-dice-value');
    const valF1Node = document.getElementById('val-f1-value');
    const perfDataWaitNode = document.getElementById('perf-data-wait');
    const perfForwardNode = document.getElementById('perf-forward');
    const perfBackwardNode = document.getElementById('perf-backward');
    const perfOptimizerNode = document.getElementById('perf-optimizer');
    const perfTotalNode = document.getElementById('perf-total');

    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const stopPanel = document.getElementById('stop-form');

    const workModeSelect = document.querySelector('[name="main-work_mode"]');
    const workModeRadios = document.querySelectorAll('input[name="work_mode_radio"]');

    const cutModeSelect = document.querySelector('[name="settings-sample_cut_mode"]');
    const cutModeRadios = document.querySelectorAll('input[name="sample_cut_mode_radio"]');

    const useValidationInput = document.querySelector('[name="settings-use_validation"]');
    const additionalProcessingInput = document.querySelector('[name="settings-additional_processing"]');

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

    const validationPercentField = document.querySelector('[data-role="validation-percent"]');
    const edgeCutField = document.querySelector('[data-role="edge-cut"]');
    const targetSizeField = document.querySelector('[data-role="target-size"]');

    let afterId = 0;

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
            const value = data[el.name];
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
        const controls = wrapper.querySelectorAll('input, select, textarea');
        controls.forEach((node) => {
            node.disabled = !enabled;
        });
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
        const mode = workModeSelect ? workModeSelect.value : 'train_and_recognition';

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
        if (mode === 'recognintion_only') {
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
        setFieldEnabled(validationPercentField, !!(useValidationInput && useValidationInput.checked));
        const enabled = !!(additionalProcessingInput && additionalProcessingInput.checked);
        setFieldEnabled(edgeCutField, enabled);
        setFieldEnabled(targetSizeField, enabled);
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

    function linePath(points, width, height, color) {
        if (!points || points.length === 0) {
            return '';
        }

        const xs = points.map((p) => Number(p.x));
        const ys = points.map((p) => Number(p.y));

        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

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
        const width = 600;
        const height = 220;
        const grid = `
            <rect x="0" y="0" width="${width}" height="${height}" fill="#0f151e"></rect>
            <line x1="0" y1="${height - 16}" x2="${width}" y2="${height - 16}" stroke="#2f3947"/>
            <line x1="16" y1="0" x2="16" y2="${height}" stroke="#2f3947"/>
        `;
        svg.innerHTML = grid + linePath(points, width, height, color);
    }

    function appendLogs(events) {
        events.forEach((event) => {
            const div = document.createElement('div');
            div.className = `log-line ${event.topic}`;
            div.textContent = `[${event.id}] ${event.message}`;
            logContainer.appendChild(div);
        });
        if (events.length > 0) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }

    function updateMetrics(metrics) {
        const trainEpoch = (metrics.train_epoch || []).map((p) => ({ x: p.epoch, y: p.loss }));
        const valEpoch = (metrics.val_epoch || []).map((p) => ({ x: p.epoch, y: p.loss }));
        const trainBatch = (metrics.train_batch || []).map((p) => ({ x: p.batch_index, y: p.loss }));
        const quality = metrics.validation_quality || {};
        const lastValPoint = valEpoch.length > 0 ? (metrics.val_epoch || [])[valEpoch.length - 1] : null;

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
        if (perfForwardNode) perfForwardNode.textContent = formatMs(perf.forward_ms);
        if (perfBackwardNode) perfBackwardNode.textContent = formatMs(perf.backward_ms);
        if (perfOptimizerNode) perfOptimizerNode.textContent = formatMs(perf.optimizer_ms);
        if (perfTotalNode) perfTotalNode.textContent = formatMs(perf.total_ms);

        drawChart(chartTrainEpoch, trainEpoch, '#58a6ff');
        drawChart(chartValEpoch, valEpoch, '#ffb86b');
        drawChart(chartTrainBatch, trainBatch, '#7ee787');
    }

    async function poll() {
        try {
            const response = await fetch(`/api/status/?after=${afterId}`, { cache: 'no-store' });
            if (!response.ok) return;

            const data = await response.json();
            const status = data.status || 'idle';
            if (statusNode) statusNode.textContent = status;
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
    if (additionalProcessingInput) additionalProcessingInput.addEventListener('change', applyDependentRules);

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

