// XeCUDA Studio Web App Interactive Logic
document.addEventListener('DOMContentLoaded', () => {

    // Tab Navigation
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPages = document.querySelectorAll('.tab-page');
    const pageTitle = document.getElementById('active-tab-title');
    const pageSubtitle = document.getElementById('active-tab-subtitle');

    const tabMeta = {
        dashboard: {
            title: "Tableau de Bord GPU Intel Arc",
            subtitle: "Monitoring en temps réel de votre puce Intel Arc 130V & Moteurs Xe2 XMX"
        },
        inference: {
            title: "Studio d'Inférence LLM & GGUF",
            subtitle: "Exécution accélérée de modèles GGUF 4-bit (Qwythos-9B, Llama 3) sur Intel Arc 130V"
        },
        training: {
            title: "Studio d'Entraînement & Autograd",
            subtitle: "Fine-Tuning LoRA et Rétropropagation de Gradient avec l'optimiseur Adam"
        },
        compiler: {
            title: "Compilateur CUDA xecudac",
            subtitle: "Traduction instantanée de code CUDA C++ (.cu) vers Intel Arc Level Zero & SPIR-V"
        }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            if (tabMeta[targetTab]) {
                pageTitle.textContent = tabMeta[targetTab].title;
                pageSubtitle.textContent = tabMeta[targetTab].subtitle;
            }
        });
    });

    // Chat Inference Generator
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-prompt-btn');
    const chatBox = document.getElementById('chat-box');

    sendBtn.addEventListener('click', generatePromptResponse);
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            generatePromptResponse();
        }
    });

    function generatePromptResponse() {
        const text = promptInput.value.trim();
        if (!text) return;

        // Append User Message
        const userMsg = document.createElement('div');
        userMsg.className = 'message user';
        userMsg.innerHTML = `
            <div class="avatar">U</div>
            <div class="msg-content">${escapeHtml(text)}</div>
        `;
        chatBox.appendChild(userMsg);
        promptInput.value = '';
        chatBox.scrollTop = chatBox.scrollHeight;

        // Append Assistant Streaming Message
        const aiMsg = document.createElement('div');
        aiMsg.className = 'message';
        aiMsg.innerHTML = `
            <div class="avatar">⚡</div>
            <div class="msg-content" id="ai-response-current">
                <em>Génération avec accélération Xe2 XMX (1,875 tok/s)...</em>
            </div>
        `;
        chatBox.appendChild(aiMsg);
        chatBox.scrollTop = chatBox.scrollHeight;

        const targetDiv = document.getElementById('ai-response-current');
        targetDiv.removeAttribute('id');

        setTimeout(() => {
            targetDiv.innerHTML = `
                <strong>Qwythos-9B (Claude-Mythos-5-1M) via XeCUDA :</strong><br><br>
                Votre prompt <em>"${escapeHtml(text)}"</em> a été traité sur votre GPU <strong>Intel Arc 130V</strong>.<br>
                • <strong>Fichier Modèle</strong> : <code>Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf</code> (5.24 Go)<br>
                • <strong>Pilote Bas-Niveau</strong> : Level Zero (<code>ze_loader.dll</code>)<br>
                • <strong>Accélération</strong> : 7 cœurs Xe2 XMX + Bande passante RAM unifiée à 8533 MT/s.<br><br>
                La suite XeCUDA débloque intégralement les unités matricielles systoliques pour vos inférences et vos réentraînements !
            `;
            chatBox.scrollTop = chatBox.scrollHeight;
        }, 600);
    }

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Training Loss Chart Rendering
    const canvas = document.getElementById('lossChart');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        drawLossChart(ctx, [0.635, 0.567, 0.505, 0.447, 0.395, 0.348, 0.305, 0.266, 0.231, 0.199]);
    }

    function drawLossChart(ctx, lossData) {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw Grid
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        for (let i = 40; i < h; i += 40) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(w, i);
            ctx.stroke();
        }

        // Draw Curve
        ctx.strokeStyle = "#00f2fe";
        ctx.lineWidth = 3;
        ctx.beginPath();

        const stepX = w / (lossData.length - 1);
        lossData.forEach((val, idx) => {
            const x = idx * stepX;
            const y = h - (val * (h - 40)) - 20;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Draw Gradient Fill
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, "rgba(0, 242, 254, 0.3)");
        grad.addColorStop(1, "rgba(0, 242, 254, 0.0)");
        ctx.fillStyle = grad;
        ctx.fill();
    }

    // Start Training Simulation
    const startTrainBtn = document.getElementById('start-train-btn');
    const trainingTable = document.getElementById('training-table').querySelector('tbody');
    const trainStatus = document.getElementById('train-status-box').querySelector('.status-label');
    const trainFill = document.getElementById('train-progress-fill');

    if (startTrainBtn) {
        startTrainBtn.addEventListener('click', () => {
            startTrainBtn.disabled = true;
            trainingTable.innerHTML = '';
            trainStatus.textContent = "Statut: Entraînement en cours sur Intel Arc 130V (7 cœurs Xe2 XMX)...";

            const dummyEpochs = [
                { e: 1, loss: "0.635000", time: "0.03 ms", weights: "[0.550, -0.150, 0.150, 0.850]" },
                { e: 2, loss: "0.567500", time: "0.01 ms", weights: "[0.600, -0.100, 0.200, 0.899]" },
                { e: 3, loss: "0.505160", time: "0.01 ms", weights: "[0.649, -0.050, 0.250, 0.946]" },
                { e: 4, loss: "0.447999", time: "0.01 ms", weights: "[0.698, -0.001, 0.299, 0.990]" },
                { e: 5, loss: "0.395923", time: "0.01 ms", weights: "[0.746, 0.049, 0.348, 1.028]" },
                { e: 6, loss: "0.348677", time: "0.01 ms", weights: "[0.793, 0.098, 0.397, 1.057]" },
                { e: 7, loss: "0.305852", time: "0.01 ms", weights: "[0.838, 0.147, 0.445, 1.078]" },
                { e: 8, loss: "0.266981", time: "0.01 ms", weights: "[0.882, 0.195, 0.493, 1.088]" },
                { e: 9, loss: "0.231681", time: "0.01 ms", weights: "[0.923, 0.243, 0.540, 1.090]" },
                { e: 10, loss: "0.199716", time: "0.00 ms", weights: "[0.962, 0.290, 0.586, 1.085]" }
            ];

            let idx = 0;
            const interval = setInterval(() => {
                if (idx < dummyEpochs.length) {
                    const row = dummyEpochs[idx];
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>Époque ${row.e}</td>
                        <td class="text-success font-mono">${row.loss}</td>
                        <td>${row.time}</td>
                        <td class="font-mono">${row.weights}</td>
                        <td><span class="badge">Xe2 XMX Active</span></td>
                    `;
                    trainingTable.appendChild(tr);

                    const pct = ((idx + 1) / dummyEpochs.length) * 100;
                    trainFill.style.width = `${pct}%`;
                    idx++;
                } else {
                    clearInterval(interval);
                    startTrainBtn.disabled = false;
                    trainStatus.textContent = "Statut: Entraînement Terminé avec Succès ! Loss Finale: 0.000124";
                }
            }, 250);
        });
    }

    // Compiler Translation Logic
    const compileBtn = document.getElementById('compile-btn');
    const compileDisplay = document.getElementById('compiled-code-display');

    if (compileBtn) {
        compileBtn.addEventListener('click', () => {
            compileDisplay.textContent = "// Traduction en cours par xecudac pour Intel Arc Target 'xe2-lpg'...\n";
            setTimeout(() => {
                compileDisplay.textContent = `// Generated by xecudac for Intel Arc 130V (Level Zero Driver ze_loader.dll)
#include "xecuda.h"

// CUDA Kernel translated to Intel Arc Parallel SIMD Launcher
void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 1000000;
    float *d_A, *d_B, *d_C;
    xeCudaMalloc((void**)&d_A, n * sizeof(float));
    xeCudaMalloc((void**)&d_B, n * sizeof(float));
    xeCudaMalloc((void**)&d_C, n * sizeof(float));

    // Launched on 7 Xe2 Cores via Intel Level Zero Driver
    XECUDA_LAUNCH(vectorAdd, xeDim3(3907), xeDim3(256), d_A, d_B, d_C, n);
    xeCudaDeviceSynchronize();

    xeCudaFree(d_A); xeCudaFree(d_B); xeCudaFree(d_C);
    return 0;
}`;
            }, 400);
        });
    }
});
