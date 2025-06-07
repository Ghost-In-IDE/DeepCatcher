// dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // Verificar si el usuario está autenticado
    checkAuthentication();
    
    // Inicializar la interfaz
    initializeDashboard();
    initializeNavigation();
    initializeVideoRecording();
    
    // Configurar logout
    document.getElementById('logoutBtn').addEventListener('click', logout);
});

function checkAuthentication() {
    // En una aplicación real, verificarías un token JWT válido
    // Por ahora, simplemente verificamos si hay datos de usuario
    if (!window.userData) {
        // Si no hay datos de usuario, redirigir al login
        window.location.href = '../login.html';
        return;
    }
}

function initializeDashboard() {
    // Mostrar información del usuario
    const userData = window.userData || { username: 'Usuario', email: 'usuario@ejemplo.com' };
    
    document.getElementById('userName').textContent = userData.username;
    document.getElementById('userEmail').textContent = userData.email;
}

function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetPage = this.getAttribute('data-page');
            
            // Remover clase active de todos los elementos
            navItems.forEach(nav => nav.classList.remove('active'));
            pages.forEach(page => page.classList.remove('active'));
            
            // Agregar clase active al elemento seleccionado
            this.classList.add('active');
            document.getElementById(targetPage + '-page').classList.add('active');
        });
    });
}

function initializeVideoRecording() {
    let mediaRecorder;
    let recordedChunks = [];
    let stream;
    let recordingTime = 0;
    let recordingInterval;
    
    const videoPreview = document.getElementById('videoPreview');
    const noCamera = document.getElementById('noCamera');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const recordingTimeDisplay = document.getElementById('recordingTime');
    const videosList = document.getElementById('videosList');
    
    // Inicializar cámara
    initCamera();
    
    async function initCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: true 
            });
            
            videoPreview.srcObject = stream;
            videoPreview.style.display = 'block';
            noCamera.style.display = 'none';
            
            // Configurar MediaRecorder
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                saveRecording();
            };
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            videoPreview.style.display = 'none';
            noCamera.style.display = 'flex';
        }
    }
    
    startBtn.addEventListener('click', function() {
        if (mediaRecorder && mediaRecorder.state === 'inactive') {
            recordedChunks = [];
            mediaRecorder.start();
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            pauseBtn.disabled = false;
            
            recordingIndicator.style.display = 'flex';
            startRecordingTimer();
        }
    });
    
    stopBtn.addEventListener('click', function() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            pauseBtn.disabled = true;
            
            recordingIndicator.style.display = 'none';
            stopRecordingTimer();
        }
    });
    
    pauseBtn.addEventListener('click', function() {
        if (mediaRecorder) {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.pause();
                pauseBtn.textContent = '▶️ Reanudar';
                stopRecordingTimer();
            } else if (mediaRecorder.state === 'paused') {
                mediaRecorder.resume();
                pauseBtn.textContent = '⏸️ Pausar';
                startRecordingTimer();
            }
        }
    });
    
    function startRecordingTimer() {
        recordingTime = 0;
        recordingTimeDisplay.textContent = '00:00';
        
        recordingInterval = setInterval(() => {
            recordingTime++;
            const minutes = Math.floor(recordingTime / 60);
            const seconds = recordingTime % 60;
            recordingTimeDisplay.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }
    
    function stopRecordingTimer() {
        if (recordingInterval) {
            clearInterval(recordingInterval);
            recordingInterval = null;
        }
    }
    
    function saveRecording() {
        if (recordedChunks.length === 0) return;
        
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const timestamp = new Date().toLocaleString();
        const nombreEntrevistado = document.getElementById('nombreEntrevistado').value || 'Sin nombre';
        
        // Crear elemento de video en la lista
        addVideoToList(url, timestamp, nombreEntrevistado, blob);
        
        // Limpiar el nombre del entrevistado
        document.getElementById('nombreEntrevistado').value = '';
        
        // Resetear tiempo
        recordingTime = 0;
        recordingTimeDisplay.textContent = '00:00';
    }
    
    function addVideoToList(url, timestamp, nombreEntrevistado, blob) {
        // Remover mensaje de "no hay videos"
        const noVideosMsg = videosList.querySelector('.no-videos');
        if (noVideosMsg) {
            noVideosMsg.remove();
        }
        
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        videoItem.innerHTML = `
            <div class="video-info">
                <h4>${nombreEntrevistado}</h4>
                <p>Fecha: ${timestamp}</p>
                <p>Duración: ${recordingTimeDisplay.textContent}</p>
            </div>
            <div class="video-actions">
                <button onclick="playVideo('${url}')" class="btn-play">▶️ Reproducir</button>
                <button onclick="downloadVideo('${url}', '${nombreEntrevistado}', '${timestamp}')" class="btn-download">💾 Descargar</button>
                <button onclick="deleteVideo(this)" class="btn-delete">🗑️ Eliminar</button>
            </div>
        `;
        
        // Guardar referencia al blob para descarga
        videoItem.dataset.blob = url;
        
        videosList.appendChild(videoItem);
    }
}

// Funciones globales para los videos
function playVideo(url) {
    const modal = document.createElement('div');
    modal.className = 'video-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <button onclick="closeModal(this)" class="close-btn">✕</button>
            <video controls autoplay style="width: 100%; max-height: 80vh;">
                <source src="${url}" type="video/webm">
            </video>
        </div>
    `;
    document.body.appendChild(modal);
}

function downloadVideo(url, nombre, timestamp) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `Entrevista_${nombre}_${timestamp.replace(/[/:]/g, '-')}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function deleteVideo(button) {
    if (confirm('¿Está seguro de que desea eliminar este video?')) {
        const videoItem = button.closest('.video-item');
        const url = videoItem.dataset.blob;
        
        // Liberar el objeto URL
        if (url) {
            URL.revokeObjectURL(url);
        }
        
        videoItem.remove();
        
        // Mostrar mensaje si no hay más videos
        const videosList = document.getElementById('videosList');
        if (videosList.children.length === 0) {
            videosList.innerHTML = '<p class="no-videos">No hay videos grabados aún</p>';
        }
    }
}

function closeModal(button) {
    const modal = button.closest('.video-modal');
    document.body.removeChild(modal);
}

function logout() {
    if (confirm('¿Está seguro de que desea cerrar sesión?')) {
        // Limpiar datos de usuario
        window.userData = null;
        
        // Redirigir al login
        window.location.href = '../login.html';
    }
}