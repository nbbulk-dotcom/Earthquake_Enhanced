/**
 * 3D Visualization using Three.js
 * Displays resonance overlay wireframe
 */

let scene, camera, renderer, controls;
let resonanceMeshes = [];

function init3DScene() {
    const container = document.getElementById('overlay3D');
    if (!container) return;
    
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    
    // Camera setup
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 50, 100);
    
    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);
    
    // Controls
    if (THREE.OrbitControls) {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
    }
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(50, 50, 50);
    scene.add(directionalLight);
    
    // Grid helper
    const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
    scene.add(gridHelper);
    
    // Start animation loop
    animate();
    
    // Handle resize
    window.addEventListener('resize', onWindowResize);
    
    console.log('3D scene initialized');
}

function animate() {
    requestAnimationFrame(animate);
    
    if (controls) {
        controls.update();
    }
    
    // Animate meshes
    resonanceMeshes.forEach(mesh => {
        if (mesh.userData.animate) {
            mesh.rotation.z += 0.01;
        }
    });
    
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById('overlay3D');
    if (!container) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    renderer.setSize(width, height);
}

function updateOverlay3D(analysisData, timeStep = 0) {
    if (!scene) {
        init3DScene();
    }
    
    // Clear existing meshes
    resonanceMeshes.forEach(mesh => {
        scene.remove(mesh);
    });
    resonanceMeshes = [];
    
    if (!analysisData || !analysisData.superposition) return;
    
    // Create visualization based on analysis data
    createResonanceVisualization(analysisData, timeStep);
}

function createResonanceVisualization(data, timeStep) {
    const superposition = data.superposition;
    const coherence = data.coherence;
    
    // Create central sphere for resultant resonance
    const geometry = new THREE.SphereGeometry(5, 32, 32);
    
    // Color based on interference type
    let color;
    switch (superposition.interference_type) {
        case 'constructive':
            color = 0xff0000;  // Red
            break;
        case 'destructive':
            color = 0x0000ff;  // Blue
            break;
        default:
            color = 0xffff00;  // Yellow
    }
    
    const material = new THREE.MeshPhongMaterial({
        color: color,
        transparent: true,
        opacity: 0.7,
        emissive: color,
        emissiveIntensity: 0.3
    });
    
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.y = superposition.resultant_amplitude * 20;
    sphere.userData.animate = true;
    scene.add(sphere);
    resonanceMeshes.push(sphere);
    
    // Create wireframe showing amplitude
    const amplitude = superposition.resultant_amplitude;
    createAmplitudeWireframe(amplitude, color);
    
    // Add frequency rings
    createFrequencyRings(superposition.resultant_frequency);
    
    // Add coherence indicators
    if (coherence.is_coherent) {
        createCoherenceIndicator(coherence.coherence_coefficient);
    }
}

function createAmplitudeWireframe(amplitude, color) {
    const radius = amplitude * 30;
    const segments = 64;
    
    const geometry = new THREE.RingGeometry(radius - 2, radius + 2, segments);
    const material = new THREE.MeshBasicMaterial({
        color: color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
    });
    
    const ring = new THREE.Mesh(geometry, material);
    ring.rotation.x = -Math.PI / 2;
    scene.add(ring);
    resonanceMeshes.push(ring);
}

function createFrequencyRings(frequency) {
    const ringCount = Math.min(Math.floor(frequency), 10);
    
    for (let i = 1; i <= ringCount; i++) {
        const geometry = new THREE.TorusGeometry(i * 10, 0.5, 16, 100);
        const material = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.3 - (i * 0.02)
        });
        
        const torus = new THREE.Mesh(geometry, material);
        torus.rotation.x = Math.PI / 2;
        scene.add(torus);
        resonanceMeshes.push(torus);
    }
}

function createCoherenceIndicator(coherence) {
    // Create pulsing sphere to indicate coherence
    const geometry = new THREE.SphereGeometry(2, 16, 16);
    const material = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        transparent: true,
        opacity: coherence * 0.8,
        wireframe: true
    });
    
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.set(30, 30, 0);
    scene.add(sphere);
    resonanceMeshes.push(sphere);
}

// Initialize when document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init3DScene);
} else {
    init3DScene();
}

// Export for global use
window.init3DScene = init3DScene;
window.updateOverlay3D = updateOverlay3D;
