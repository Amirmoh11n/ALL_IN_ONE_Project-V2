const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const fileInput = $('fileInput');
const preview = $('preview');
const fileName = $('fileName');
const analyzeBtn = $('analyzeBtn');
let selectedFile = null;

function showError(message) { $('errorBox').textContent = message; $('errorBox').classList.remove('hidden'); }
function clearError() { $('errorBox').classList.add('hidden'); }
function selectFile(file) {
  clearError();
  if (!file) return;
  const allowed = ['image/png','image/jpeg','image/webp','image/bmp'];
  if (!allowed.includes(file.type)) return showError('Please choose a PNG, JPG, WEBP, or BMP image.');
  if (file.size > 10 * 1024 * 1024) return showError('The selected image is larger than 10 MB.');
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  $('emptyState').classList.add('hidden'); $('previewState').classList.remove('hidden');
  $('resultEmpty').classList.remove('hidden'); $('resultState').classList.add('hidden');
}

dropzone.addEventListener('click', (e) => { if (e.target !== analyzeBtn) fileInput.click(); });
fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
['dragenter','dragover'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.add('drop-active'); }));
['dragleave','drop'].forEach(type => dropzone.addEventListener(type, e => { e.preventDefault(); dropzone.classList.remove('drop-active'); }));
dropzone.addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));

async function checkHealth() {
  try {
    const r = await fetch('/api/health'); const data = await r.json();
    $('statusDot').className = `h-2 w-2 rounded-full ${data.model_ready ? 'bg-emerald-400' : 'bg-amber-400'}`;
    $('statusText').textContent = data.model_ready ? 'Model ready' : 'Model not exported';
  } catch { $('statusDot').className = 'h-2 w-2 rounded-full bg-red-400'; $('statusText').textContent = 'API unavailable'; }
}

analyzeBtn.addEventListener('click', async (e) => {
  e.stopPropagation(); clearError();
  if (!selectedFile) return;
  analyzeBtn.disabled = true; analyzeBtn.textContent = 'Analyzing…';
  try {
    const form = new FormData(); form.append('file', selectedFile);
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Inference failed.');
    $('resultEmpty').classList.add('hidden'); $('resultState').classList.remove('hidden');
    $('predictedClass').textContent = data.predicted_class;
    $('confidence').textContent = `${data.confidence_percentage.toFixed(1)}%`;
    $('confidenceBar').style.width = `${data.confidence_percentage}%`;
    $('modelBadge').textContent = 'Completed'; $('device').textContent = data.device.replace('ExecutionProvider',''); $('warning').textContent = data.warning || '';
    $('probabilities').innerHTML = data.ranked_probabilities.map(item => `
      <div><div class="mb-1 flex justify-between text-xs"><span class="capitalize text-slate-300">${item.class_name}</span><span class="text-slate-500">${item.percentage.toFixed(1)}%</span></div><div class="h-1.5 overflow-hidden rounded-full bg-white/10"><div class="bar h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400" style="width:${item.percentage}%"></div></div></div>`).join('');
  } catch (err) { showError(err.message); }
  finally { analyzeBtn.disabled = false; analyzeBtn.textContent = 'Analyze MRI'; }
});
checkHealth();
