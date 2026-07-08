const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const feedbackBox = document.getElementById('feedbackBox');
const predictionMeta = document.getElementById('predictionMeta');
const modelSwitcher = document.getElementById('modelSwitcher');
const switcherHint = document.getElementById('switcherHint');

const MAX_BYTES = 16 * 1024 * 1024;

let selectedFile = null;
let currentModel = 'deep';
let lastData = null;

const MODEL_HINTS = {
    naive: 'Majority-class baseline',
    classical: 'Random Forest + 50 handcrafted features',
    deep: 'MobileNetV2 multi-task',
};

modelSwitcher.addEventListener('click', (e) => {
    const btn = e.target.closest('.model-btn');
    if (!btn) return;
    currentModel = btn.dataset.model;
    document.querySelectorAll('.model-btn').forEach(b => b.classList.toggle('active', b === btn));
    switcherHint.textContent = MODEL_HINTS[currentModel] || '';
    if (lastData) displayResults(lastData);
});

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    } else {
        const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
        if (url) loadSampleAsFile(url);
    }
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showFeedback('error', 'Invalid file type. JPG or PNG only.', null);
        return;
    }
    if (file.size > MAX_BYTES) {
        showFeedback('error', `File too large (${(file.size/1024/1024).toFixed(1)}MB). 16MB max.`, null);
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        uploadContent.style.display = 'none';
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    const formData = new FormData();
    formData.append('image', selectedFile);
    const t0 = performance.now();
    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
        data._inference_ms = performance.now() - t0;
        displayResults(data);
    } catch (err) {
        showFeedback('error', `Prediction failed: ${err.message}`, null);
        resultsSection.style.display = 'block';
    } finally {
        predictBtn.disabled = false;
        loading.style.display = 'none';
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';
    lastData = data;
    const naive = data.naive || {};
    const classical = data.classical || {};
    const deep = data.deep || {};
    const sources = { naive, classical, deep };
    const source = sources[currentModel] && sources[currentModel].car_type
        ? sources[currentModel]
        : (deep.car_type ? deep : classical);
    if (source.car_type) updateCard('CarType', source.car_type);
    if (source.door_count) updateCard('DoorCount', source.door_count);
    if (source.seat_count) updateCard('SeatCount', source.seat_count);

    const primaryConf = source.car_type ? source.car_type.confidence : 0;
    renderFeedback(primaryConf, data.feedback);
    renderTopK(data.top_k || []);
    renderExplanations(data.explanations || []);
    renderComparison(naive, classical, deep);

    const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
    const ms = data._inference_ms !== undefined ? ` · ${data._inference_ms.toFixed(0)}ms` : '';
    predictionMeta.textContent = `${currentModel}${ms} · ${ts}`;

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateCard(suffix, result) {
    document.getElementById('res' + suffix).textContent = result.prediction;
    document.getElementById('conf' + suffix).textContent = `conf ${(result.confidence * 100).toFixed(1)}%`;
    const probsDiv = document.getElementById('probs' + suffix);
    probsDiv.innerHTML = '';
    const probs = result.probabilities || {};
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 4);
    sorted.forEach(([label, prob]) => {
        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.innerHTML = `
            <div class="prob-label"><span>${label}</span><span class="pct">${(prob*100).toFixed(1)}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:${prob*100}%"></div></div>
        `;
        probsDiv.appendChild(bar);
    });
}

function renderFeedback(confidence, feedback) {
    const level = (feedback && feedback.level) || 'info';
    const message = (feedback && feedback.message) || 'Done.';
    const icon = level === 'success' ? '✓' : level === 'warning' ? '!' : level === 'error' ? '×' : 'i';
    feedbackBox.className = `feedback-box feedback-${level}`;
    feedbackBox.innerHTML = `
        <span class="feedback-icon">${icon}</span>
        <span class="feedback-message">${message}</span>
        <span class="feedback-conf">${confidence ? (confidence*100).toFixed(1) + '%' : '-'}</span>
    `;
}

function showFeedback(level, message, conf) {
    const icon = level === 'success' ? '✓' : level === 'warning' ? '!' : level === 'error' ? '×' : 'i';
    feedbackBox.className = `feedback-box feedback-${level}`;
    feedbackBox.innerHTML = `
        <span class="feedback-icon">${icon}</span>
        <span class="feedback-message">${message}</span>
        <span class="feedback-conf">${conf !== null && conf !== undefined ? conf : '-'}</span>
    `;
}

function renderTopK(topK) {
    const list = document.getElementById('topkList');
    list.innerHTML = '';
    if (!topK.length) {
        list.innerHTML = '<div class="topk-row"><span class="topk-rank">-</span><span class="topk-label">No top-k data (deep model not loaded)</span><span class="topk-bar"></span><span class="topk-conf">-</span></div>';
        return;
    }
    topK.forEach((item, i) => {
        const row = document.createElement('div');
        row.className = 'topk-row';
        const label = item.label || item.prediction || '-';
        const prob = item.probability || item.confidence || 0;
        row.innerHTML = `
            <span class="topk-rank">${String(i+1).padStart(2, '0')}</span>
            <span class="topk-label">${label}</span>
            <div class="topk-bar"><div class="topk-fill" style="width:${prob*100}%"></div></div>
            <span class="topk-conf">${(prob*100).toFixed(1)}%</span>
        `;
        list.appendChild(row);
    });
}

function renderExplanations(exps) {
    const list = document.getElementById('explanationList');
    list.innerHTML = '';
    if (!exps.length) {
        list.innerHTML = '<div class="explanation-item"><span class="idx">-</span><span class="text">No feature breakdown available.</span></div>';
        return;
    }
    exps.forEach((exp, i) => {
        const text = typeof exp === 'string' ? exp : (exp.text || JSON.stringify(exp));
        const item = document.createElement('div');
        item.className = 'explanation-item';
        item.innerHTML = `<span class="idx">${String(i+1).padStart(2, '0')}</span><span class="text">${text}</span>`;
        list.appendChild(item);
    });
}

function renderComparison(naive, classical, deep) {
    const grid = document.getElementById('comparisonGrid');
    const tasks = [
        ['car_type', 'Car Type'],
        ['door_count', 'Doors'],
        ['seat_count', 'Seats'],
    ];
    const hasDeep = deep && deep.car_type;
    let rows = tasks.map(([key, label]) => {
        const n = (naive[key] || {});
        const c = (classical[key] || {});
        const d = (deep[key] || {});
        const nPred = n.prediction || '-';
        const cPred = c.prediction || '-';
        const dPred = d.prediction || '-';
        const nConf = n.confidence !== undefined ? `${(n.confidence*100).toFixed(1)}%` : '-';
        const cConf = c.confidence !== undefined ? `${(c.confidence*100).toFixed(1)}%` : '-';
        const dConf = d.confidence !== undefined ? `${(d.confidence*100).toFixed(1)}%` : '-';
        const preds = [nPred, cPred, dPred].filter(p => p !== '-');
        const disagree = hasDeep && new Set(preds).size > 1;
        return `
            <div class="cmp-row ${disagree ? 'cmp-disagree' : ''}">
                <div class="cmp-label">${label}${disagree ? '<span class="cmp-flag">DISAGREE</span>' : ''}</div>
                <div class="cmp-cell cmp-naive">
                    <span class="cmp-pred">${nPred}</span>
                    <span class="cmp-conf">${nConf}</span>
                </div>
                <div class="cmp-cell cmp-classical">
                    <span class="cmp-pred">${cPred}</span>
                    <span class="cmp-conf">${cConf}</span>
                </div>
                <div class="cmp-cell cmp-deep">
                    <span class="cmp-pred">${dPred}</span>
                    <span class="cmp-conf">${dConf}</span>
                </div>
            </div>
        `;
    }).join('');
    grid.innerHTML = `
        <div class="cmp-table">
            <div class="cmp-head">
                <div class="cmp-label">TASK</div>
                <div class="cmp-cell">Naive</div>
                <div class="cmp-cell">Classical · RF</div>
                <div class="cmp-cell">Deep · MobileNetV2</div>
            </div>
            ${rows}
        </div>
    `;
}

async function loadSampleAsFile(url) {
    try {
        const resp = await fetch(url);
        const blob = await resp.blob();
        const name = url.split('/').pop() || 'sample.jpg';
        const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
        handleFile(file);
    } catch (err) {
        showFeedback('error', `Could not load sample: ${err.message}`, null);
    }
}

async function loadSamples() {
    try {
        const res = await fetch('/samples');
        const data = await res.json();
        const grid = document.getElementById('samplesGrid');
        grid.innerHTML = '';
        (data.samples || []).forEach(s => {
            const tile = document.createElement('div');
            tile.className = 'sample-tile';
            tile.draggable = true;
            tile.dataset.url = s.url;
            tile.dataset.filename = s.filename || '';
            tile.innerHTML = `
                <img src="${s.url}" alt="${s.label}" loading="lazy" draggable="false">
                <span class="label">${s.label}</span>
            `;
            tile.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/uri-list', s.url);
                e.dataTransfer.setData('text/plain', s.url);
                e.dataTransfer.effectAllowed = 'copy';
                tile.classList.add('dragging');
            });
            tile.addEventListener('dragend', () => tile.classList.remove('dragging'));
            tile.addEventListener('click', () => loadSampleAsFile(s.url));
            grid.appendChild(tile);
        });
    } catch (e) {
        console.warn('samples load failed', e);
    }
}

loadSamples();
