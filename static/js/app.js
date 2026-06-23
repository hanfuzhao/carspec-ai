// CarSpec AI — 前端交互逻辑
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results');

let selectedFile = null;

// 上传区域交互
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
        alert('请上传图片文件');
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

// 预测
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    predictBtn.disabled = true;
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    const formData = new FormData();
    formData.append('image', selectedFile);
    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        displayResults(data);
    } catch (err) {
        alert('预测失败: ' + err.message);
    } finally {
        predictBtn.disabled = false;
        loading.style.display = 'none';
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';
    // Classical 结果
    const classical = data.classical || {};
    const deep = data.deep || {};
    // 优先显示 Deep，回退 Classical
    const source = deep || classical;
    if (source.car_type) updateCard('CarType', source.car_type);
    if (source.door_count) updateCard('DoorCount', source.door_count);
    if (source.seat_count) updateCard('SeatCount', source.seat_count);
    // 可解释说明
    const expList = document.getElementById('explanationList');
    expList.innerHTML = '';
    (data.explanations || []).forEach(exp => {
        const item = document.createElement('div');
        item.className = 'explanation-item';
        item.innerHTML = `<span class="icon">▸</span><span>${exp}</span>`;
        expList.appendChild(item);
    });
    // 模型对比
    const compGrid = document.getElementById('comparisonGrid');
    compGrid.innerHTML = `
        <div class="comparison-col">
            <h4>Classical ML (随机森林)</h4>
            <div class="comparison-result">${classical.car_type ? classical.car_type.prediction : '—'}</div>
            ${classical.car_type ? `<div style="color:var(--success);font-size:14px">置信度 ${(classical.car_type.confidence*100).toFixed(1)}%</div>` : ''}
        </div>
        <div class="comparison-col">
            <h4>Deep Learning (ResNet50)</h4>
            <div class="comparison-result">${deep.car_type ? deep.car_type.prediction : '—'}</div>
            ${deep.car_type ? `<div style="color:var(--success);font-size:14px">置信度 ${(deep.car_type.confidence*100).toFixed(1)}%</div>` : ''}
        </div>
    `;
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function updateCard(suffix, result) {
    document.getElementById('res' + suffix).textContent = result.prediction;
    document.getElementById('conf' + suffix).textContent = `置信度 ${(result.confidence * 100).toFixed(1)}%`;
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
