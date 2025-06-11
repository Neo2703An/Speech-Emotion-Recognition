document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const audioUpload = document.getElementById('audio-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const selectedFileContainer = document.querySelector('.selected-file-container');
    const removeFileBtn = document.getElementById('remove-file');
    const analyzeBtn = document.getElementById('analyze-btn');
    const audioPlayer = document.querySelector('.audio-player');
    const audioElement = document.getElementById('audio-element');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const playIcon = document.getElementById('play-icon');
    const pauseIcon = document.getElementById('pause-icon');
    const resultsSection = document.querySelector('.results-section');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    const emotionEmoji = document.getElementById('emotion-emoji');
    const emotionName = document.getElementById('emotion-name');
    const modelName = document.getElementById('model-name');
    const confidenceLevel = document.querySelector('.confidence-level');
    const confidenceValue = document.getElementById('confidence-value');
    const modelRadios = document.querySelectorAll('input[name="model"]');
    const currentTimeDisplay = document.getElementById('current-time');
    const totalDurationDisplay = document.getElementById('total-duration');
    const frequencyDisplay = document.getElementById('frequency-display');
    const amplitudeDisplay = document.getElementById('amplitude-display');
    
    // Variables
    let selectedFile = null;
    let isPlaying = false;
    let selectedModel = 'cnn'; // Default model
    let lastAnalyzedFile = null;
    let lastAnalyzedModel = null;
    let wavesurfer = null;
    let audioContext = null;
    let audioAnalyser = null;
    let frequencyData = null;
    let timeData = null;
    let frequencyBars = [];
    let animationFrameId = null;
    
    // Initialize WaveSurfer
    function initWaveSurfer() {
        if (wavesurfer) {
            wavesurfer.destroy();
        }
        
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#4776E6',
            progressColor: '#8E54E9',
            cursorColor: 'transparent',
            barWidth: 2,
            barGap: 1,
            height: 80,
            barRadius: 2,
            responsive: true,
            normalize: true,
            partialRender: true
        });
        
        // WaveSurfer events
        wavesurfer.on('ready', function() {
            // Update duration display
            const duration = wavesurfer.getDuration();
            totalDurationDisplay.textContent = formatTime(duration);
            
            // Enable play button
            playPauseBtn.disabled = false;
        });
        
        wavesurfer.on('audioprocess', function() {
            // Update current time display
            const currentTime = wavesurfer.getCurrentTime();
            currentTimeDisplay.textContent = formatTime(currentTime);
        });
        
        wavesurfer.on('play', function() {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
            isPlaying = true;
            
            // Start visualization if audio context exists
            if (audioContext && audioAnalyser) {
                updateVisualization();
            }
        });
        
        wavesurfer.on('pause', function() {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
            isPlaying = false;
            
            // Stop visualization
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        });
        
        wavesurfer.on('finish', function() {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
            isPlaying = false;
            
            // Stop visualization
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        });
    }
    
    // Initialize audio visualization
    function initAudioVisualization() {
        // Create audio context
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Create analyser node
        audioAnalyser = audioContext.createAnalyser();
        audioAnalyser.fftSize = 256;
        
        // Connect audio element to analyser
        const source = audioContext.createMediaElementSource(audioElement);
        source.connect(audioAnalyser);
        audioAnalyser.connect(audioContext.destination);
        
        // Create data arrays
        frequencyData = new Uint8Array(audioAnalyser.frequencyBinCount);
        timeData = new Uint8Array(audioAnalyser.frequencyBinCount);
        
        // Create frequency bars
        frequencyDisplay.innerHTML = '';
        const barCount = 32; // Number of frequency bars to display
        for (let i = 0; i < barCount; i++) {
            const bar = document.createElement('div');
            bar.className = 'frequency-bar';
            bar.style.height = '0px';
            frequencyDisplay.appendChild(bar);
            frequencyBars.push(bar);
        }
        
        // Create amplitude line
        amplitudeDisplay.innerHTML = '';
        const centerLine = document.createElement('div');
        centerLine.className = 'amplitude-line';
        amplitudeDisplay.appendChild(centerLine);
        
        // Create SVG for amplitude wave
        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("class", "amplitude-wave");
        svg.appendChild(path);
        amplitudeDisplay.appendChild(svg);
    }
    
    // Update visualization
    function updateVisualization() {
        if (!audioAnalyser || !isPlaying) return;
        
        // Get frequency data
        audioAnalyser.getByteFrequencyData(frequencyData);
        
        // Update frequency bars
        const barCount = frequencyBars.length;
        for (let i = 0; i < barCount; i++) {
            // Get frequency value (0-255)
            const index = Math.floor(i * audioAnalyser.frequencyBinCount / barCount);
            const value = frequencyData[index];
            
            // Scale to bar height (0-150px)
            const height = (value / 255) * 140;
            frequencyBars[i].style.height = `${height}px`;
        }
        
        // Get time domain data
        audioAnalyser.getByteTimeDomainData(timeData);
        
        // Draw amplitude wave
        const svgWidth = amplitudeDisplay.clientWidth;
        const svgHeight = amplitudeDisplay.clientHeight;
        const path = amplitudeDisplay.querySelector('.amplitude-wave');
        
        let d = `M 0 ${svgHeight / 2}`;
        const sliceWidth = svgWidth / timeData.length;
        
        for (let i = 0; i < timeData.length; i++) {
            const v = timeData[i] / 128.0;
            const y = v * svgHeight / 2;
            d += ` L ${i * sliceWidth} ${y}`;
        }
        
        path.setAttribute("d", d);
        
        // Continue animation
        animationFrameId = requestAnimationFrame(updateVisualization);
    }
    
    // Format time in seconds to MM:SS
    function formatTime(seconds) {
        seconds = Math.floor(seconds);
        const minutes = Math.floor(seconds / 60);
        seconds = seconds % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
    
    // Event Listeners
    audioUpload.addEventListener('change', handleFileSelect);
    removeFileBtn.addEventListener('click', removeFile);
    analyzeBtn.addEventListener('click', analyzeAudio);
    playPauseBtn.addEventListener('click', togglePlayPause);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    
    // Add event listeners to model radio buttons
    modelRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            selectedModel = this.value;
            console.log(`Selected model: ${selectedModel}`);
            
            // If the file is the same but model changed, enable analyze button
            if (selectedFile && (lastAnalyzedFile !== selectedFile || lastAnalyzedModel !== selectedModel)) {
                analyzeBtn.disabled = false;
            }
        });
    });
    
    // Handle file selection
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            selectedFile = file;
            fileNameDisplay.textContent = file.name;
            selectedFileContainer.style.display = 'block';
            analyzeBtn.disabled = false;
            
            // Initialize WaveSurfer
            initWaveSurfer();
            
            // Create audio URL
            const audioURL = URL.createObjectURL(file);
            audioElement.src = audioURL;
            
            // Load audio into WaveSurfer
            wavesurfer.load(audioURL);
            
            // Show audio player
            audioPlayer.style.display = 'block';
            
            // Hide results if visible
            resultsSection.style.display = 'none';
        }
    }
    
    // Remove selected file
    function removeFile() {
        selectedFile = null;
        lastAnalyzedFile = null;
        fileNameDisplay.textContent = 'No file selected';
        selectedFileContainer.style.display = 'none';
        analyzeBtn.disabled = true;
        audioPlayer.style.display = 'none';
        
        // Reset audio
        if (wavesurfer) {
            wavesurfer.pause();
            wavesurfer.empty();
        }
        
        audioElement.pause();
        audioElement.src = '';
        
        // Reset time display
        currentTimeDisplay.textContent = '0:00';
        totalDurationDisplay.textContent = '0:00';
        
        // Stop visualization
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }
    
    // Toggle play/pause
    function togglePlayPause() {
        if (wavesurfer) {
            wavesurfer.playPause();
        }
    }
    
    // Analyze audio
    function analyzeAudio() {
        if (!selectedFile) return;
        
        // Disable analyze button during analysis
        analyzeBtn.disabled = true;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Get selected model
        const selectedModel = document.querySelector('input[name="model"]:checked').value;
        
        // Remember the file and model being analyzed
        lastAnalyzedFile = selectedFile;
        lastAnalyzedModel = selectedModel;
        
        // Create form data
        const formData = new FormData();
        formData.append('audio', selectedFile);
        formData.append('model', selectedModel);
        
        // Send to backend
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
            
            // Display results
            displayResults(data);
            
            // Initialize audio visualization
            initAudioVisualization();
            
            // Keep analyze button disabled since we've already analyzed this file with this model
            analyzeBtn.disabled = true;
        })
        .catch(error => {
            console.error('Error:', error);
            loadingOverlay.style.display = 'none';
            alert('An error occurred during analysis. Please try again.');
            
            // Re-enable analyze button on error
            analyzeBtn.disabled = false;
        });
    }
    
    // Display results
    function displayResults(data) {
        // Set emotion emoji
        const emoji = getEmotionEmoji(data.emotion);
        emotionEmoji.textContent = emoji;
        
        // Set emotion name
        emotionName.textContent = capitalizeFirstLetter(data.emotion);
        
        // Set model name
        modelName.textContent = data.model.toUpperCase();
        
        // Set confidence
        const confidencePercent = (data.confidence * 100).toFixed(1);
        confidenceLevel.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Display visualizations if available
        if (data.visualizations) {
            // Get visualization containers
            const frequencyVisualization = document.getElementById('frequency-visualization');
            const amplitudeVisualization = document.getElementById('amplitude-visualization');
            
            // Create visualization tabs
            const visualizationContainer = document.querySelector('.visualization-container');
            visualizationContainer.innerHTML = '';
            
            // Add mel spectrogram visualization
            if (data.visualizations.mel_spectrogram) {
                const melSpectrogramDiv = document.createElement('div');
                melSpectrogramDiv.className = 'visualization-item';
                melSpectrogramDiv.innerHTML = `
                    <h4>Mel Spectrogram</h4>
                    <div class="visualization-display">
                        <img src="data:image/png;base64,${data.visualizations.mel_spectrogram}" alt="Mel Spectrogram" class="visualization-image">
                    </div>
                `;
                visualizationContainer.appendChild(melSpectrogramDiv);
            }
            
            // Add MFCC visualization
            if (data.visualizations.mfcc) {
                const mfccDiv = document.createElement('div');
                mfccDiv.className = 'visualization-item';
                mfccDiv.innerHTML = `
                    <h4>MFCC</h4>
                    <div class="visualization-display">
                        <img src="data:image/png;base64,${data.visualizations.mfcc}" alt="MFCC" class="visualization-image">
                    </div>
                `;
                visualizationContainer.appendChild(mfccDiv);
            }
            
            // Add waveform visualization
            if (data.visualizations.waveform) {
                const waveformDiv = document.createElement('div');
                waveformDiv.className = 'visualization-item';
                waveformDiv.innerHTML = `
                    <h4>Waveform</h4>
                    <div class="visualization-display">
                        <img src="data:image/png;base64,${data.visualizations.waveform}" alt="Waveform" class="visualization-image">
                    </div>
                `;
                visualizationContainer.appendChild(waveformDiv);
            }
        }
        
        // Show results section
        resultsSection.style.display = 'block';
    }
    
    // Reset analysis for new sample
    function resetAnalysis() {
        // Hide results
        resultsSection.style.display = 'none';
        
        // Reset file selection
        removeFile();
        
        // Focus on file upload
        audioUpload.click();
    }
    
    // Helper functions
    function getEmotionEmoji(emotion) {
        const emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
            'fear': '😨',
            'disgust': '🤢',
            'surprise': '😯'
        };
        return emojis[emotion] || '😐';
    }
    
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Add drag and drop functionality
    const uploadLabel = document.querySelector('.upload-label');
    
    uploadLabel.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('drag-over');
    });
    
    uploadLabel.addEventListener('dragleave', function() {
        this.classList.remove('drag-over');
    });
    
    uploadLabel.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            audioUpload.files = e.dataTransfer.files;
            const event = new Event('change');
            audioUpload.dispatchEvent(event);
        }
    });
}); 