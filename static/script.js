import { initializeApp } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.10.0/firebase-analytics.js";

const firebaseConfig = {
  apiKey: "AIzaSyCqT78SBBMFoy6ri5yHOiXY777jSqvM00w",
  authDomain: "ai-vigilance.firebaseapp.com",
  projectId: "ai-vigilance",
  storageBucket: "ai-vigilance.firebasestorage.app",
  messagingSenderId: "258696912295",
  appId: "1:258696912295:web:b21063a4963084f882ea82",
  measurementId: "G-8GFEFVQNTZ"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

document.addEventListener('DOMContentLoaded', () => {

    // -------------------------------------------------------------------------
    // 1. Register Person
    // -------------------------------------------------------------------------
    const regForm = document.getElementById('register-form');
    if (regForm) {
        regForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const status = document.getElementById('reg-status');
            status.innerText = "Registering...";
            try {
                const res = await fetch('/register', { method: 'POST', body: new FormData(regForm) });
                const data = await res.json();
                status.innerText = data.message;
                if (data.status === 'success') regForm.reset();
            } catch { status.innerText = "Error registering."; }
        });
    }

    // -------------------------------------------------------------------------
    // 1.a People page — list all registered persons
    // -------------------------------------------------------------------------
    const personsList = document.getElementById('persons-list');
    if (personsList) {
        const loadPersons = async () => {
            try {
                const res = await fetch('/api/persons');
                const persons = await res.json();
                if (!persons || persons.length === 0) {
                    personsList.innerHTML = '<p style="color:#666;">No registered persons yet.</p>';
                } else {
                    personsList.innerHTML = persons.map(p => `
                        <div class="detection-card" style="text-align:center;">
                            <img class="detection-image" src="/${p.image_path}" alt="${p.name}" 
                                 onerror="this.src='/static/default-avatar.png'">
                            <div class="detection-info">
                                <h4>${p.name}</h4>
                            </div>
                        </div>
                    `).join('');
                }
            } catch {
                personsList.innerHTML = '<p style="color:#ff3333;">Failed to load persons.</p>';
            }
        };
        loadPersons();
    }

    // -------------------------------------------------------------------------
    // 2. Camera type dropdown placeholder
    // -------------------------------------------------------------------------
    const camTypeDropdown = document.getElementById('camera-type');
    const camSourceInput = document.getElementById('camera-source');
    if (camTypeDropdown && camSourceInput) {
        const placeholders = {
            webcam: 'Camera Index (e.g. 0)',
            rtsp: 'e.g. rtsp://user:pass@ip:554',
            ipwebcam: 'e.g. 192.168.1.100  (port auto-appended)',
            droidcam: 'e.g. 192.168.1.100  (port 4747 auto-appended)'
        };
        camTypeDropdown.addEventListener('change', (e) => {
            camSourceInput.placeholder = placeholders[e.target.value] || '';
        });
    }

    // -------------------------------------------------------------------------
    // 3. Add Camera form
    // -------------------------------------------------------------------------
    const camForm = document.getElementById('camera-form');
    if (camForm) {
        camForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const res = await fetch('/add_camera', { method: 'POST', body: new FormData(camForm) });
            const data = await res.json();
            if (data.status === 'success') location.reload();
            else alert(data.message || 'Failed to connect camera.');
        });
    }

    // -------------------------------------------------------------------------
    // 4. Manage cameras list
    // -------------------------------------------------------------------------
    const cameraList = document.getElementById('active-cameras-list');
    if (cameraList) {
        const fetchCameras = async () => {
            const res = await fetch('/api/cameras');
            const cams = await res.json();
            cameraList.innerHTML = cams.map(cam => `
                <li style="display:flex;justify-content:space-between;align-items:center;
                           background:#222;padding:10px;margin-bottom:5px;border-radius:5px;">
                    <span>${cam}</span>
                    <button onclick="deleteCamera('${cam}')"
                        style="background:#ff3333;color:white;border:none;padding:5px 10px;
                               cursor:pointer;border-radius:3px;width:auto;margin:0;">Remove</button>
                </li>`).join('');
            const occSelect = document.getElementById('occ-camera-select');
            if (occSelect) {
                const current = occSelect.value;
                occSelect.innerHTML = '<option value=\"\">All Cameras</option>' + cams.map(c => `<option value="${c}">${c}</option>`).join('');
                if (current) occSelect.value = current;
            }
        };
        fetchCameras();
        setInterval(fetchCameras, 5000);
    }

    const occBtn = document.getElementById('occ-load-btn');
    const occTbody = document.getElementById('occ-tbody');
    if (occBtn && occTbody) {
        const loadOcc = async () => {
            const cam = document.getElementById('occ-camera-select')?.value || '';
            const start = document.getElementById('occ-start')?.value || '';
            const end = document.getElementById('occ-end')?.value || '';
            const params = new URLSearchParams();
            if (cam) params.append('camera_id', cam);
            if (start) params.append('start_time', start);
            if (end) params.append('end_time', end);
            const res = await fetch(`/api/occupancy?${params.toString()}`);
            const rows = await res.json();
            if (!rows || rows.length === 0) {
                occTbody.innerHTML = '<tr><td style="padding:8px;color:#666;" colspan="3">No data</td></tr>';
                return;
            }
            occTbody.innerHTML = rows.map(r => `
                <tr style="border-bottom:1px solid #222;">
                    <td style="padding:8px;">${new Date(r.timestamp).toLocaleString()}</td>
                    <td style="padding:8px;">${r.camera_id}</td>
                    <td style="padding:8px;">${r.count}</td>
                </tr>
            `).join('');
        };
        occBtn.addEventListener('click', loadOcc);
        loadOcc();
    }
    // -------------------------------------------------------------------------
    // 5. Search Page — Active search mission + history grid
    // -------------------------------------------------------------------------
    const missionBanner = document.getElementById('mission-banner');
    const missionText   = document.getElementById('mission-text');
    const stopBtn       = document.getElementById('stop-btn');
    const searchBtn     = document.getElementById('search-btn');
    const searchImgBtn  = document.getElementById('search-image-btn');
    const clearBtn      = document.getElementById('clear-btn');
    const grid          = document.getElementById('results-grid');
    const targetPreview = document.getElementById('target-preview');
    const targetPhoto   = document.getElementById('target-photo');
    const targetName    = document.getElementById('target-name');

    // Render detection cards
    const renderGrid = (data) => {
        if (!grid) return;
        if (!data || data.length === 0) {
            grid.innerHTML = '<p style="color:#666;padding:1rem;">No detections found.</p>';
            return;
        }
        grid.innerHTML = data.map(d => `
            <div class="detection-card">
                <img class="detection-image" src="/${d.image_path}" alt="snap"
                     onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'160\\' height=\\'120\\'><rect fill=\\'%23333\\' width=\\'100%25\\' height=\\'100%25\\'/><text fill=\\'%23aaa\\' x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dy=\\'.3em\\'>No Image</text></svg>'">
                <div class="detection-info">
                    <h4>${d.person_name}</h4>
                    <p>📍 ${d.camera_id}</p>
                    <p>🕒 ${new Date(d.timestamp).toLocaleString()}</p>
                </div>
            </div>`).join('');
    };

    // Fetch history by name + date filters
    const fetchHistory = async (name = '') => {
        const start = document.getElementById('start-time')?.value || '';
        const end   = document.getElementById('end-time')?.value || '';
        const params = new URLSearchParams({ name, start_time: start, end_time: end });
        const res = await fetch(`/api/search?${params}`);
        renderGrid(await res.json());
    };

    // Check whether a search is already running on page load
    const syncMissionBanner = async () => {
        const res = await fetch('/api/active_search');
        const data = await res.json();
        if (data.active && missionBanner) {
            missionBanner.classList.remove('hidden');
            if (missionText) missionText.textContent = `🔴 Actively searching for: ${data.name}`;
            if (targetPreview) targetPreview.classList.remove('hidden');
            if (targetName) targetName.textContent = data.name;
            fetchHistory(data.name);
        }
    };
    if (missionBanner) syncMissionBanner();

    // Start Search — triggers live mission AND loads history
    if (searchBtn) {
        searchBtn.addEventListener('click', async () => {
            const name = document.getElementById('name-search')?.value?.trim();
            if (!name) { alert('Please enter a name.'); return; }

            searchBtn.disabled = true;
            searchBtn.innerText = 'Starting...';

            const fd = new FormData();
            fd.append('name', name);
            const res  = await fetch('/api/start_search', { method: 'POST', body: fd });
            const data = await res.json();

            if (data.status === 'success') {
                if (missionBanner) { missionBanner.classList.remove('hidden'); }
                if (missionText)   { missionText.textContent = `🔴 Actively searching for: ${name}`; }
                // Show target person photo from DB
                if (targetPreview) targetPreview.classList.remove('hidden');
                if (targetName)    targetName.textContent = data.name;
                if (targetPhoto && data.image_path) {
                    targetPhoto.src = '/' + data.image_path;
                }
                fetchHistory(name); // Show existing history below
            } else {
                alert(data.message || 'Person not registered.');
            }

            searchBtn.disabled = false;
            searchBtn.innerText = '🔍 Start Search';
        });
    }

    // Stop Search
    if (stopBtn) {
        stopBtn.addEventListener('click', async () => {
            await fetch('/api/stop_search', { method: 'POST' });
            if (missionBanner) missionBanner.classList.add('hidden');
            if (targetPreview) targetPreview.classList.add('hidden');
        });
    }

    // Search by Image
    if (searchImgBtn) {
        searchImgBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('image-search-input');
            if (!fileInput || fileInput.files.length === 0) { alert('Select an image first.'); return; }
            const fd = new FormData();
            fd.append('file', fileInput.files[0]);
            searchImgBtn.disabled = true;
            searchImgBtn.innerText = 'Searching...';
            const res = await fetch('/api/search_by_image', { method: 'POST', body: fd });
            renderGrid(await res.json());
            searchImgBtn.disabled = false;
            searchImgBtn.innerText = '📷 Search by Photo';
        });
    }

    // Clear History
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            if (!confirm('Clear all detection history? This cannot be undone.')) return;
            clearBtn.disabled = true;
            clearBtn.innerText = 'Clearing...';
            try {
                await fetch('/clear_history', { method: 'POST' });
                renderGrid([]);
                clearBtn.innerText = '✓ Cleared!';
                clearBtn.style.background = '#2ecc71';
                setTimeout(() => {
                    clearBtn.innerText = '🗑 Clear History';
                    clearBtn.style.background = '';
                    clearBtn.disabled = false;
                }, 3000);
            } catch {
                clearBtn.innerText = '🗑 Clear History';
                clearBtn.disabled = false;
            }
        });
    }

    // Initial load on search page (all history)
    if (grid && !missionBanner) fetchHistory();
    if (grid && missionBanner) {
        // Already handled by syncMissionBanner; load all if no active search
        fetch('/api/active_search').then(r => r.json()).then(d => {
            if (!d.active) fetchHistory();
        });
    }
});

// -------------------------------------------------------------------------
// Global helpers
// -------------------------------------------------------------------------
window.deleteCamera = async (camId) => {
    if (!confirm(`Remove camera "${camId}"?`)) return;
    const fd = new FormData();
    fd.append('camera_id', camId);
    await fetch('/delete_camera', { method: 'POST', body: fd });
    location.reload();
};

window.toggleRecording = async (camId, btn) => {
    btn.disabled = true;
    const fd = new FormData();
    fd.append("camera_id", camId);
    try {
        const res = await fetch("/api/toggle_recording", { method: "POST", body: fd });
        const data = await res.json();
        if (data.status === "success") {
            if (data.recording) {
                btn.innerHTML = "⏹ Stop Rec";
                btn.style.background = "#555";
            } else {
                btn.innerHTML = "⏺ Record";
                btn.style.background = "#ff3333";
            }
        } else {
            alert(data.message || "Failed to toggle recording.");
        }
    } catch {
        alert("Network error.");
    }
    btn.disabled = false;
};

// Sync active recording buttons on load:
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/recording_status');
        const data = await res.json();
        if (data.active_recordings) {
            data.active_recordings.forEach(cam => {
                const btn = document.querySelector(`button[data-rec-cam="${cam}"]`);
                if (btn) {
                    btn.innerHTML = "⏹ Stop Rec";
                    btn.style.background = "#555";
                }
            });
        }
    } catch (e) {}
});
