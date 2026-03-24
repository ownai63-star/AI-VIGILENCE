document.addEventListener('DOMContentLoaded', () => {
    // 1. Handle Registration
    const regForm = document.getElementById('register-form');
    if (regForm) {
        regForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(regForm);
            const status = document.getElementById('reg-status');
            status.innerText = "Registering...";
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                status.innerText = result.message;
                if (result.status === "success") {
                    regForm.reset();
                }
            } catch (err) {
                status.innerText = "Error registering.";
            }
        });
    }

    // 2. Handle Add Camera
    const camTypeDropdown = document.getElementById('camera-type');
    const camSourceInput = document.getElementById('camera-source');
    
    if (camTypeDropdown && camSourceInput) {
        camTypeDropdown.addEventListener('change', (e) => {
            const type = e.target.value;
            if (type === 'webcam') {
                camSourceInput.placeholder = "Camera Index (e.g. 0)";
            } else if (type === 'rtsp') {
                camSourceInput.placeholder = "e.g. rtsp://test:dei@12@12@10.7.16.48:554";
            } else if (type === 'ipwebcam') {
                camSourceInput.placeholder = "e.g. http://192.168.1.100:8080/video";
            } else if (type === 'droidcam') {
                camSourceInput.placeholder = "e.g. http://192.168.1.100:4747/mjpegfeed";
            }
        });
    }

    const camForm = document.getElementById('camera-form');
    if (camForm) {
        camForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(camForm);
            try {
                const response = await fetch('/add_camera', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (result.status === "success") {
                  location.reload();
                } else {
                  alert("Failed to connect camera.");
                }
            } catch (err) {
                alert("Error connecting camera.");
            }
        });
    }

    // 3. Handle Search
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', async () => {
            if(confirm("Clear all detection history? This cannot be undone.")) {
                clearBtn.disabled = true;
                clearBtn.innerText = "Clearing...";
                try {
                    const resp = await fetch('/clear_history', { method: 'POST' });
                    const result = await resp.json();
                    // Immediately empty the results grid
                    const grid = document.getElementById('results-grid');
                    if (grid) {
                        grid.innerHTML = '<p style="color:#aaa; padding: 1rem;">History cleared. No detections yet.</p>';
                    }
                    clearBtn.innerText = "✓ Cleared!";
                    clearBtn.style.background = '#2ecc71';
                    setTimeout(() => {
                        clearBtn.innerText = "Clear History";
                        clearBtn.style.background = '#ff3333';
                        clearBtn.disabled = false;
                    }, 3000);
                } catch(e) {
                    alert("Error clearing history.");
                    clearBtn.disabled = false;
                    clearBtn.innerText = "Clear History";
                }
            }
        });
    }

    const searchBtn = document.getElementById('search-btn');
    const searchImgBtn = document.getElementById('search-image-btn');
    
    if (searchBtn || searchImgBtn) {
        const renderData = (data) => {
            const grid = document.getElementById('results-grid');
            if (grid) {
                if (data.length === 0) {
                    grid.innerHTML = "<p>No results found.</p>";
                    return;
                }
                grid.innerHTML = data.map(d => `
                    <div class="detection-card">
                        <img class="detection-image" src="/${d.image_path}" alt="Snap">
                        <div class="detection-info">
                            <h4>${d.person_name}</h4>
                            <p>📍 ${d.camera_id}</p>
                            <p>🕒 ${new Date(d.timestamp).toLocaleString()}</p>
                        </div>
                    </div>
                `).join('');
            }
        };

        const fetchTextDetections = async () => {
            const name = document.getElementById('name-search').value;
            const start = document.getElementById('start-time').value;
            const end = document.getElementById('end-time').value;
            
            const params = new URLSearchParams({ name, start_time: start, end_time: end });
            const response = await fetch(`/api/search?${params}`);
            renderData(await response.json());
        };

        const fetchImageDetections = async () => {
            const fileInput = document.getElementById('image-search-input');
            if(fileInput.files.length === 0) {
                alert("Please select an image first.");
                return;
            }
            const fd = new FormData();
            fd.append("file", fileInput.files[0]);
            
            const response = await fetch('/api/search_by_image', {
                method: 'POST',
                body: fd
            });
            renderData(await response.json());
        };

        if(searchBtn) searchBtn.addEventListener('click', fetchTextDetections);
        if(searchImgBtn) searchImgBtn.addEventListener('click', fetchImageDetections);
        
        if(searchBtn) fetchTextDetections(); // Initial load
    }
    
    // 4. Handle Manage Cameras
    const cameraList = document.getElementById('active-cameras-list');
    if (cameraList) {
        const fetchCameras = async () => {
            const response = await fetch('/api/cameras');
            const cameras = await response.json();
            
            cameraList.innerHTML = cameras.map(cam => `
                <li style="display: flex; justify-content: space-between; align-items: center; background: #222; padding: 10px; margin-bottom: 5px; border-radius: 5px;">
                    <span>${cam}</span>
                    <button onclick="deleteCamera('${cam}')" style="background: #ff3333; color: white; border: none; padding: 5px 10px; cursor: pointer; border-radius: 3px; width: auto; margin: 0;">Remove</button>
                </li>
            `).join('');
        };
        fetchCameras();
    }
});

// Global function for onclick
window.deleteCamera = async (camId) => {
    if(confirm(`Remove ${camId}?`)){
        const fd = new FormData();
        fd.append('camera_id', camId);
        await fetch('/delete_camera', {method: 'POST', body: fd});
        location.reload();
    }
};
