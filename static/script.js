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
        };
        fetchCameras();
        setInterval(fetchCameras, 5000);
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
