document.addEventListener('DOMContentLoaded', () => {
    const switchBtn = document.getElementById('switch-btn');
    const modelSelect = document.getElementById('model-select');
    const loadingOverlay = document.getElementById('loading-overlay');
    const liveStream = document.getElementById('live-stream');
    
    // Modal elements
    const metricsBtn = document.getElementById('metrics-btn');
    const metricsModal = document.getElementById('metrics-modal');
    const closeModal = document.getElementById('close-modal');
    const metricsContent = document.getElementById('metrics-content');

    let currentModel = 'yolo26';

    // Handle Model Switch
    switchBtn.addEventListener('click', async () => {
        const selectedModel = modelSelect.value;
        if (selectedModel === currentModel) return;

        // Show loading state
        loadingOverlay.classList.remove('hidden');
        switchBtn.disabled = true;
        switchBtn.innerText = "Switching...";

        try {
            const response = await fetch(`/api/switch_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: selectedModel })
            });
            const data = await response.json();
            
            if (data.status === 'success') {
                currentModel = selectedModel;
                // Force reload image feed cache to drop stutter
                liveStream.src = `/video_feed?t=${new Date().getTime()}`;
            } else {
                alert("Error switching model: " + data.message);
                modelSelect.value = currentModel;
            }
        } catch (err) {
            alert("Network error: " + err);
            modelSelect.value = currentModel;
        } finally {
            loadingOverlay.classList.add('hidden');
            switchBtn.disabled = false;
            switchBtn.innerText = "Switch Model";
        }
    });

    // Handle Metrics Modal Opening
    metricsBtn.addEventListener('click', async () => {
        metricsModal.classList.remove('hidden');
        metricsContent.innerHTML = `<p style="text-align:center">Loading metrics for ${currentModel}...</p>`;
        
        try {
            const response = await fetch(`/api/metrics/${currentModel}`);
            const data = await response.json();
            
            let html = `
                <div class="model-desc">
                    <p><strong>Architecture logic:</strong> ${data.architecture}</p>
                    <p>${data.description}</p>
                </div>
                <table class="metrics-table">
                    <tbody>
                        <tr><th>Device Backend</th><td>${data.device}</td></tr>
                        <tr><th>Target Dataset</th><td>${data.dataset}</td></tr>
                        <tr><th>Baseline COCO mAP</th><td>${data.mAP}</td></tr>
                        <tr><th>Estimated Parameters</th><td>${data.params}</td></tr>
                    </tbody>
                </table>
            `;
            metricsContent.innerHTML = html;
        } catch (err) {
            metricsContent.innerHTML = `<p style="color:red">Failed to load metrics.</p>`;
        }
    });

    // Close Modal
    closeModal.addEventListener('click', () => {
        metricsModal.classList.add('hidden');
    });

    // Click outside to close modal
    metricsModal.addEventListener('click', (e) => {
        if (e.target === metricsModal) {
            metricsModal.classList.add('hidden');
        }
    });
});
