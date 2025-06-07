// login.js
document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const errorMessage = document.getElementById('error-message');

    // Credenciales de ejemplo (en un caso real, esto sería validado en el servidor)
    const validCredentials = {
        'admin': 'admin123',
        'usuario': 'password',
        'test': 'test123'
    };

    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const usuario = document.getElementById('usuario').value.trim();
        const contraseña = document.getElementById('contraseña').value.trim();

        // Limpiar mensaje de error previo
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';

        // Validar que los campos no estén vacíos
        if (!usuario || !contraseña) {
            showError('Por favor, complete todos los campos');
            return;
        }

        // Validar credenciales
        if (validCredentials[usuario] && validCredentials[usuario] === contraseña) {
            // Login exitoso
            // Guardar información del usuario
            const userData = {
                username: usuario,
                email: usuario + '@deepcatcher.com',
                loginTime: new Date().toISOString()
            };
            
            // Guardar en memoria (en una aplicación real usarías tokens JWT)
            window.userData = userData;
            
            // Redirigir al dashboard
            window.location.href = 'web/dashboard.html';
        } else {
            showError('Usuario o contraseña incorrectos');
        }
    });

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        
        // Limpiar el mensaje después de 5 segundos
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }

    // Limpiar errores cuando el usuario empiece a escribir
    document.getElementById('usuario').addEventListener('input', function() {
        if (errorMessage.style.display === 'block') {
            errorMessage.style.display = 'none';
        }
    });

    document.getElementById('contraseña').addEventListener('input', function() {
        if (errorMessage.style.display === 'block') {
            errorMessage.style.display = 'none';
        }
    });
});