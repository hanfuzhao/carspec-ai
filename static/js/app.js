const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results');
const feedbackBox = document.getElementById('feedbackBox');
const topkList = document.getElementById('topkList');
const samplesGrid = document.getElementById('samplesGrid');

let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (JPG, PNG, BMP, or WEBP).');
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum allowed size is 16MB.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        uploadContent.style.display = 'none';
        predictBtn.disabled = false;
        hideError();
    };
    reader.readAsDataURL(file);
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    hideError();
    const formData = new FormData();
    formData.append('image', selectedFile);
    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || `HTTP ${res.status}`);
        }
        displayResults(data);
    } catch (err) {
        showError('Prediction failed: ' + err.message);
    } finally {
        predictBtn.disabled = false;
        loading.style.display = 'none';
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';
    const classical = data.classical || {};
    const deep = data.deep || {};
    const source = deep || classical;

    renderFeedback(data.feedback);

    if (source.car_type) updateCard('CarType', source.car_type);
    if (source.door_count) updateCard('DoorCount', source.door_count);
    if (source.seat_count) updateCard('SeatCount', source.seat_count);

    const expList = document.getElementById('explanationList');
    expList.innerHTML = '';
    (data.explanations || []).forEach(exp => {
        const item = document.createElement('div');
        item.className = 'explanation-item';
        item.innerHTML = `<span class="icon">▸</span><span>${exp}</span>`;
        expList.appendChild(item);
    });

    const compGrid = document.getElementById('comparisonGrid');
    compGrid.innerHTML = `
        <div class="comparison-col">
            <h4>Classical ML (Random Forest)</h4>
            <div class="comparison-result">${classical.car_type ? classical.car_type.prediction : '—'}</div>
            ${classical.car_type ? `<div style="color:var(--success);font-size:14px">Confidence ${(classical.car_type.confidence*100).toFixed(1)}%</div>` : ''}
        </div>
        <div class="comparison-col">
            <h4>Deep Learning (MobileNetV2)</h4>
            <div class="comparison-result">${deep.car_type ? deep.car_type.prediction : '—'}</div>
            ${deep.car_type ? `<div style="color:var(--success);font-size:14px">Confidence ${(deep.car_type.confidence*100).toFixed(1)}%</div>` : ''}
        </div>
    `;

    renderTopK(data.top_k || []);
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderFeedback(feedback) {
    if (!feedback) {
        feedbackBox.style.display = 'none';
        return;
    }
    feedbackBox.style.display = 'block';
    feedbackBox.className = `feedback-box feedback-${feedback.level}`;
    feedbackBox.innerHTML = `
        <span class="feedback-icon">${getFeedbackIcon(feedback.level)}</span>
        <span class="feedback-message">${feedback.message}</span>
    `;
}

function getFeedbackIcon(level) {
    const icons = { success: '✓', warning: '!', error: '✕', info: 'i' };
    return icons[level] || 'i';
}

function renderTopK(topK) {
    topkList.innerHTML = '';
    if (!topK || topK.length === 0) return;
    topK.forEach((item, idx) => {
        const row = document.createElement('div');
        row.className = 'topk-row';
        row.innerHTML = `
            <div class="topk-rank">#${idx + 1}</div>
            <div class="topk-label">${item.label}</div>
            <div class="topk-bar"><div class="topk-fill" style="width:${item.confidence * 100}%"></div></div>
            <div class="topk-conf">${(item.confidence * 100).toFixed(1)}%</div>
        `;
        topkList.appendChild(row);
    });
}

function showError(msg) {
    let errBox = document.getElementById('errorBox');
    if (!errBox) {
        errBox = document.createElement('div');
        errBox.id = 'errorBox';
        errBox.className = 'feedback-box feedback-error';
        uploadArea.parentNode.insertBefore(errBox, uploadArea.nextSibling);
    }
    errBox.style.display = 'block';
    errBox.className = 'feedback-box feedback-error';
    errBox.innerHTML = `<span class="feedback-icon">✕</span><span class="feedback-message">${msg}</span>`;
}

function hideError() {
    const errBox = document.getElementById('errorBox');
    if (errBox) errBox.style.display = 'none';
}

function updateCard(suffix, result) {
    document.getElementById('res' + suffix).textContent = result.prediction;
    document.getElementById('conf' + suffix).textContent = `Confidence ${(result.confidence * 100).toFixed(1)}%`;
    const probsDiv = document.getElementById('probs' + suffix);
    probsDiv.innerHTML = '';
    const probs = result.probabilities || {};
    const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([label, prob]) => {
        const bar = document.createElement('div');
        bar.className = 'prob-bar';
        bar.innerHTML = `
            <div class="prob-label"><span>${label}</span><span>${(prob*100).toFixed(1)}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:${prob*100}%"></div></div>
        `;
        probsDiv.appendChild(bar);
    });
}

async function loadSamples() {
    try {
        const res = await fetch('/samples');
        const data = await res.json();
        if (!data.samples || data.samples.length === 0) return;
        samplesGrid.innerHTML = '';
        data.samples.forEach(name => {
            const tile = document.createElement('div');
            tile.className = 'sample-tile';
            tile.innerHTML = `<img src="/static/samples/${name}" alt="${name}" loading="lazy">`;
            tile.addEventListener('click', async () => {
                const blob = await fetch(`/static/samples/${name}`).then(r => r.blob());
                const file = new File([blob], name, { type: blob.type });
                handleFile(file);
            });
            samplesGrid.appendChild(tile);
        });
    } catch (e) {
        console.warn('Samples load failed:', e);
    }
}

document.addEventListener('DOMContentLoaded', loadSamples);
