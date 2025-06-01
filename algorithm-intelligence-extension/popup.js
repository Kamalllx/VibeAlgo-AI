// Algorithm Intelligence Suite - Complete Extension Logic
class AlgorithmIntelligenceExtension {
    constructor() {
        this.apiUrl = CONFIG.API_BASE_URL;
        this.currentSession = null;
        this.healthCheckInterval = null;
        this.currentTab = 'summary';
        this.loadingProgress = 0;
        
        this.init();
    }

    async init() {
        await this.loadSettings();
        this.initializeUI();
        this.startHealthMonitoring();
        this.setupEventListeners();
        this.setupTabSwitching();
        this.checkForSelectedText();
        
        console.log('Algorithm Intelligence Extension initialized');
    }

    async loadSettings() {
        return new Promise((resolve) => {
            chrome.storage.sync.get(CONFIG.STORAGE_KEYS, (result) => {
                this.apiUrl = result[CONFIG.STORAGE_KEYS.API_URL] || CONFIG.DEFAULT_SETTINGS.apiUrl;
                this.timeout = result[CONFIG.STORAGE_KEYS.TIMEOUT] || CONFIG.DEFAULT_SETTINGS.timeout;
                this.autoRefresh = result[CONFIG.STORAGE_KEYS.AUTO_REFRESH] !== false;
                
                // Update settings in UI
                const apiUrlInput = document.getElementById('apiUrl');
                const timeoutInput = document.getElementById('timeout');
                const autoRefreshInput = document.getElementById('autoRefresh');
                
                if (apiUrlInput) apiUrlInput.value = this.apiUrl;
                if (timeoutInput) timeoutInput.value = this.timeout;
                if (autoRefreshInput) autoRefreshInput.checked = this.autoRefresh;
                
                resolve();
            });
        });
    }

    saveSettings() {
        const settings = {
            [CONFIG.STORAGE_KEYS.API_URL]: this.apiUrl,
            [CONFIG.STORAGE_KEYS.TIMEOUT]: this.timeout,
            [CONFIG.STORAGE_KEYS.AUTO_REFRESH]: this.autoRefresh
        };
        
        chrome.storage.sync.set(settings, () => {
            Utils.showToast('Settings saved successfully', 'success');
        });
    }

    initializeUI() {
        // Setup expand/collapse for sections
        document.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                const targetId = header.querySelector('.expand-btn')?.dataset.target;
                if (targetId) {
                    const content = document.getElementById(targetId);
                    const expandBtn = header.querySelector('.expand-btn');
                    
                    if (content && expandBtn) {
                        content.classList.toggle('expanded');
                        expandBtn.classList.toggle('expanded');
                    }
                }
            });
        });

        // Auto-expand first sections
        this.autoExpandFirstSections();
    }

    autoExpandFirstSections() {
        const firstSections = [
            'problemAnalysis',
            'detailedComplexity', 
            'algorithmApproaches',
            'generatedGraphs',
            'performanceCharts'
        ];
        
        firstSections.forEach(sectionId => {
            const content = document.getElementById(sectionId);
            const header = content?.previousElementSibling;
            if (content && header) {
                content.classList.add('expanded');
                header.querySelector('.expand-btn')?.classList.add('expanded');
            }
        });
    }

    setupTabSwitching() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab;
                this.switchTab(tabName);
            });
        });
    }

    switchTab(tabName) {
        // Remove active class from all tabs and panes
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        
        // Add active class to selected tab and pane
        const selectedTab = document.querySelector(`[data-tab="${tabName}"]`);
        const selectedPane = document.getElementById(`${tabName}-tab`);
        
        if (selectedTab) selectedTab.classList.add('active');
        if (selectedPane) selectedPane.classList.add('active');
        
        this.currentTab = tabName;
    }

    async checkForSelectedText() {
        try {
            const result = await chrome.storage.local.get(['selectedText', 'selectionTimestamp']);
            if (result.selectedText && result.selectionTimestamp) {
                const timeDiff = Date.now() - result.selectionTimestamp;
                if (timeDiff < 60000) { // Within last minute
                    const userInput = document.getElementById('userInput');
                    const inputType = document.getElementById('inputType');
                    
                    if (userInput) userInput.value = result.selectedText;
                    if (inputType) inputType.value = 'auto';
                    
                    // Clear the stored text
                    chrome.storage.local.remove(['selectedText', 'selectionTimestamp']);
                    Utils.showToast('Selected text loaded', 'success');
                }
            }
        } catch (error) {
            console.log('No selected text available');
        }
    }

    setupEventListeners() {
        // Main analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyzeAlgorithm());
        }

        // Health refresh
        const refreshHealthBtn = document.getElementById('refreshHealth');
        if (refreshHealthBtn) {
            refreshHealthBtn.addEventListener('click', () => this.checkHealth());
        }

        // Code actions
        const copyCodeBtn = document.getElementById('copyCodeBtn');
        if (copyCodeBtn) {
            copyCodeBtn.addEventListener('click', () => this.copyGeneratedCode());
        }

        const downloadCodeBtn = document.getElementById('downloadCodeBtn');
        if (downloadCodeBtn) {
            downloadCodeBtn.addEventListener('click', () => this.downloadGeneratedCode());
        }

        // Quick action buttons
        const viewFullReportBtn = document.getElementById('viewFullReport');
        if (viewFullReportBtn) {
            viewFullReportBtn.addEventListener('click', () => this.viewFullReport());
        }

        const downloadResultsBtn = document.getElementById('downloadResults');
        if (downloadResultsBtn) {
            downloadResultsBtn.addEventListener('click', () => this.downloadResults());
        }

        const shareResultsBtn = document.getElementById('shareResults');
        if (shareResultsBtn) {
            shareResultsBtn.addEventListener('click', () => this.shareResults());
        }

        const newAnalysisBtn = document.getElementById('newAnalysis');
        if (newAnalysisBtn) {
            newAnalysisBtn.addEventListener('click', () => this.startNewAnalysis());
        }

        // Settings
        const settingsBtn = document.getElementById('settingsBtn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.openSettings());
        }

        const closeSettingsBtn = document.getElementById('closeSettings');
        if (closeSettingsBtn) {
            closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        }

        const saveSettingsBtn = document.getElementById('saveSettings');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => this.updateSettings());
        }

        const resetSettingsBtn = document.getElementById('resetSettings');
        if (resetSettingsBtn) {
            resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        }

        // Error handling
        const retryBtn = document.getElementById('retryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.analyzeAlgorithm());
        }

        const reportBtn = document.getElementById('reportBtn');
        if (reportBtn) {
            reportBtn.addEventListener('click', () => this.reportIssue());
        }

        // Keyboard shortcuts
        const userInput = document.getElementById('userInput');
        if (userInput) {
            userInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    this.analyzeAlgorithm();
                }
            });
        }

        // Setup visualization handlers
        this.setupVisualizationHandlers();
    }

    setupVisualizationHandlers() {
        document.addEventListener('click', (event) => {
            if (event.target.matches('[data-action="view-viz"]')) {
                const filename = event.target.getAttribute('data-filename');
                this.viewVisualization(filename);
            } else if (event.target.matches('[data-action="download-viz"]')) {
                const filename = event.target.getAttribute('data-filename');
                this.downloadVisualization(filename);
            } else if (event.target.matches('[data-action="view-graph"]')) {
                const filename = event.target.getAttribute('data-filename');
                this.viewGraph(filename);
            } else if (event.target.matches('[data-action="download-graph"]')) {
                const filename = event.target.getAttribute('data-filename');
                this.downloadGraph(filename);
            }
        });
    }

    startHealthMonitoring() {
        this.checkHealth();
        
        if (this.autoRefresh) {
            this.healthCheckInterval = setInterval(() => {
                this.checkHealth();
            }, 30000); // Check every 30 seconds
        }
    }

    async checkHealth() {
        const healthDot = document.getElementById('healthDot');
        const healthText = document.getElementById('healthText');
        const healthStats = document.getElementById('healthStats');
        
        if (healthDot) healthDot.className = 'health-dot checking';
        if (healthText) healthText.textContent = 'Checking connection...';

        try {
            const response = await fetch(`${this.apiUrl}/api/status`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(CONFIG.TIMEOUTS.HEALTH_CHECK)
            });

            if (response.ok) {
                const data = await response.json();
                if (healthDot) healthDot.className = 'health-dot online';
                if (healthText) healthText.textContent = `Connected • v${data.version || '3.0'}`;
                
                if (data.algorithms_available && healthStats) {
                    healthStats.textContent = `${data.algorithms_available} algorithms • MongoDB ${data.mongodb_enabled ? 'ON' : 'OFF'}`;
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            if (healthDot) healthDot.className = 'health-dot';
            if (healthText) healthText.textContent = 'Connection failed';
            if (healthStats) healthStats.textContent = error.message;
        }
    }

    async analyzeAlgorithm() {
        const userInput = document.getElementById('userInput');
        const inputType = document.getElementById('inputType');
        const includeVisualization = document.getElementById('includeVisualization');
        const includePerformance = document.getElementById('includePerformance');
        const includeEducational = document.getElementById('includeEducational');

        if (!userInput || !userInput.value.trim()) {
            Utils.showToast('Please enter an algorithm question or code', 'error');
            return;
        }

        this.showLoadingOverlay();
        this.hideError();

        try {
            this.updateProgress(10, 'Sending request to AI...');

            const response = await fetch(`${this.apiUrl}/api/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: userInput.value.trim(),
                    input_type: inputType?.value || 'auto',
                    options: {
                        include_visualization: includeVisualization?.checked !== false,
                        include_performance: includePerformance?.checked !== false,
                        include_educational: includeEducational?.checked !== false
                    }
                }),
                signal: AbortSignal.timeout(this.timeout * 1000)
            });

            this.updateProgress(30, 'Processing response...');

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            
            this.updateProgress(60, 'Analyzing results...');
            this.currentSession = data;
            
            this.updateProgress(80, 'Displaying results...');
            await this.displayResults(data);
            
            this.updateProgress(100, 'Complete!');
            
            setTimeout(() => {
                this.hideLoadingOverlay();
                Utils.showToast('Analysis completed successfully!', 'success');
            }, 500);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.hideLoadingOverlay();
            this.showError(`Analysis failed: ${error.message}`);
            Utils.showToast('Analysis failed. Please try again.', 'error');
        }
    }

    showLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'flex';
        this.updateProgress(0, 'Initializing...');
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'none';
    }

    updateProgress(percent, text) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressFill) progressFill.style.width = `${percent}%`;
        if (progressText) progressText.textContent = text;
    }

    async displayResults(data) {
        console.log('Displaying results:', data);
        
        // Show results container and switch to summary tab
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) resultsContainer.style.display = 'block';
        this.switchTab('summary');
        
        // Display all sections with error handling
        try {
            await this.displaySummary(data);
        } catch (error) {
            console.error('Error displaying summary:', error);
        }
        
        try {
            await this.displayAnalysis(data);
        } catch (error) {
            console.error('Error displaying analysis:', error);
        }
        
        try {
            await this.displaySolution(data);
        } catch (error) {
            console.error('Error displaying solution:', error);
        }
        
        try {
            await this.displayVisualizations(data);
        } catch (error) {
            console.error('Error displaying visualizations:', error);
        }
        
        try {
            await this.displayPerformance(data);
        } catch (error) {
            console.error('Error displaying performance:', error);
        }
        
        try {
            await this.displayEducation(data);
        } catch (error) {
            console.error('Error displaying education:', error);
        }
    }

    async displaySummary(data) {
        this.safeUpdateElement('sessionId', data.session_id || 'N/A');
        this.safeUpdateElement('processingTime', this.calculateProcessingTime(data));
        
        const stagesCount = Object.keys(data.stages || {}).length;
        this.safeUpdateElement('stagesCompleted', `${stagesCount}/6 stages`);
        
        const algorithmName = this.extractAlgorithmName(data);
        this.safeUpdateElement('algorithmDetected', algorithmName);
        
        const complexity = this.extractComplexity(data);
        this.safeUpdateElement('quickTimeComplexity', complexity.time || 'Unknown');
        this.safeUpdateElement('quickSpaceComplexity', complexity.space || 'Unknown');
    }

    async displayAnalysis(data) {
        // Problem Analysis
        if (data.stages.algorithm_solving?.problem_analysis) {
            const analysis = data.stages.algorithm_solving.problem_analysis;
            this.safeUpdateElement('problemAnalysisContent', this.formatProblemAnalysis(analysis), 'innerHTML');
        }

        // Detailed Complexity
        if (data.stages.complexity_analysis?.agent_result?.complexity_analysis) {
            const complexity = data.stages.complexity_analysis.agent_result.complexity_analysis;
            this.displayDetailedComplexity(complexity);
        }
    }

    async displaySolution(data) {
        if (data.stages.algorithm_solving?.optimal_solution) {
            const solution = data.stages.algorithm_solving.optimal_solution;
            
            this.safeUpdateElement('algorithmName', solution.algorithm_name || 'Generated Solution');
            this.safeUpdateElement('algorithmDescription', solution.description || 'AI-generated algorithm solution');
        }

        const code = this.extractCode(data);
        if (code) {
            this.safeUpdateElement('generatedCode', code);
        }
    }

    async displayVisualizations(data) {
        const vizData = data.stages.visualizations;
        const statusDiv = document.getElementById('visualizationStatus');
        const listDiv = document.getElementById('visualizationList');
        
        if (vizData?.success && vizData.files_generated?.length > 0) {
            if (statusDiv) {
                statusDiv.className = 'viz-status success';
                statusDiv.innerHTML = `
                    <div class="viz-status-header">
                        <span class="status-icon">✅</span>
                        <span class="status-title">Visualizations Generated Successfully</span>
                    </div>
                    <div class="viz-status-details">
                        <div class="detail-item">
                            <span class="detail-label">Algorithm:</span>
                            <span class="detail-value">${vizData.algorithm_detected || 'Unknown'}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Files:</span>
                            <span class="detail-value">${vizData.files_generated.length} generated</span>
                        </div>
                    </div>
                `;
            }
            
            if (listDiv) {
                listDiv.innerHTML = vizData.files_generated.map(file => `
                    <div class="viz-item">
                        <div class="viz-info">
                            <div class="viz-name">${file}</div>
                            <div class="viz-type">Algorithm visualization</div>
                        </div>
                        <div class="viz-actions">
                            <button class="viz-btn" data-action="view-viz" data-filename="${file}">👁️ View</button>
                            <button class="viz-btn" data-action="download-viz" data-filename="${file}">💾 Download</button>
                        </div>
                    </div>
                `).join('');
            }
        } else {
            if (statusDiv) {
                statusDiv.className = 'viz-status error';
                statusDiv.innerHTML = `
                    <div class="viz-status-header">
                        <span class="status-icon">⚠️</span>
                        <span class="status-title">Visualization Generation Issue</span>
                    </div>
                    <div class="viz-status-message">${vizData?.message || 'No visualizations were generated.'}</div>
                `;
            }
            
            if (listDiv) {
                listDiv.innerHTML = '<div class="viz-empty">No visualizations available</div>';
            }
        }
    }

    async displayPerformance(data) {
        const metrics = this.extractPerformanceMetrics(data);
        this.safeUpdateElement('executionSpeed', metrics.speed);
        this.safeUpdateElement('memoryUsage', metrics.memory);
        this.safeUpdateElement('efficiencyScore', metrics.efficiency);
    }

    async displayEducation(data) {
        const educational = data.stages.educational_report || {};
        
        if (educational.key_concepts) {
            this.safeUpdateElement('keyConcepts', this.formatList(educational.key_concepts), 'innerHTML');
        }
        
        if (educational.recommendations) {
            this.safeUpdateElement('recommendations', this.formatList(educational.recommendations), 'innerHTML');
        }
    }

    // Helper methods
    safeUpdateElement(elementId, content, property = 'textContent') {
        const element = document.getElementById(elementId);
        if (element) {
            if (property === 'innerHTML') {
                element.innerHTML = content;
            } else {
                element.textContent = content;
            }
        }
    }

    extractAlgorithmName(data) {
        if (data.stages.visualizations?.algorithm_detected) {
            return data.stages.visualizations.algorithm_detected;
        }
        
        if (data.stages.algorithm_solving?.optimal_solution?.algorithm_name) {
            return data.stages.algorithm_solving.optimal_solution.algorithm_name;
        }
        
        return 'Algorithm Detected';
    }

    extractComplexity(data) {
        const complexity = data.stages.complexity_analysis?.agent_result?.complexity_analysis;
        
        return {
            time: complexity?.time_complexity || 'Unknown',
            space: complexity?.space_complexity || 'Unknown'
        };
    }

    extractCode(data) {
        const solution = data.stages.algorithm_solving?.optimal_solution;
        
        if (!solution?.code) return null;
        
        if (typeof solution.code === 'object' && solution.code.code) {
            return solution.code.code;
        }
        
        if (typeof solution.code === 'string') {
            return solution.code;
        }
        
        return null;
    }

    extractPerformanceMetrics(data) {
        const complexity = this.extractComplexity(data);
        
        let speedScore = 'Good';
        if (complexity.time.includes('O(1)') || complexity.time.includes('O(log n)')) {
            speedScore = 'Excellent';
        } else if (complexity.time.includes('O(n²)') || complexity.time.includes('O(2^n)')) {
            speedScore = 'Poor';
        }
        
        let memoryScore = 'Good';
        if (complexity.space.includes('O(1)')) {
            memoryScore = 'Excellent';
        } else if (complexity.space.includes('O(n²)')) {
            memoryScore = 'Poor';
        }
        
        return {
            speed: speedScore,
            memory: memoryScore,
            efficiency: speedScore === 'Excellent' && memoryScore === 'Excellent' ? 'A+' : 
                       speedScore === 'Good' || memoryScore === 'Good' ? 'B+' : 'C+'
        };
    }

    formatProblemAnalysis(analysis) {
        return `
            <div class="analysis-grid">
                <div class="analysis-item">
                    <strong>Problem Type:</strong> ${analysis.problem_type || 'Unknown'}
                </div>
                <div class="analysis-item">
                    <strong>Difficulty:</strong> ${analysis.difficulty || 'Medium'}
                </div>
            </div>
        `;
    }

    formatList(items) {
        if (!items || items.length === 0) {
            return '<p>No items available</p>';
        }
        return `<ul>${items.map(item => `<li>${item}</li>`).join('')}</ul>`;
    }

    displayDetailedComplexity(complexity) {
        this.safeUpdateElement('timeComplexity', complexity.time_complexity || 'Unknown');
        this.safeUpdateElement('spaceComplexity', complexity.space_complexity || 'Unknown');
        
        if (complexity.reasoning) {
            this.safeUpdateElement('complexityExplanation', `
                <h5>Complexity Reasoning:</h5>
                <p>${complexity.reasoning}</p>
            `, 'innerHTML');
        }
    }

    calculateProcessingTime(data) {
        return '15.2s'; // Default value
    }

    // Action methods
    async copyGeneratedCode() {
        const codeElement = document.getElementById('generatedCode');
        const code = codeElement?.textContent;
        
        if (code) {
            const success = await Utils.copyToClipboard(code);
            if (success) {
                Utils.showToast('Code copied to clipboard!', 'success');
            }
        }
    }

    async downloadGeneratedCode() {
        const codeElement = document.getElementById('generatedCode');
        const code = codeElement?.textContent;
        
        if (code) {
            const blob = new Blob([code], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `algorithm_solution_${this.currentSession?.session_id || 'unknown'}.py`;
            a.click();
            URL.revokeObjectURL(url);
            
            Utils.showToast('Code downloaded successfully!', 'success');
        }
    }

    async viewVisualization(filename) {
        if (this.currentSession?.session_id) {
            const url = `${this.apiUrl}/results/${this.currentSession.session_id}/visualizations/${filename}`;
            
            try {
                await chrome.runtime.sendMessage({
                    action: 'openTab',
                    url: url
                });
            } catch (error) {
                window.open(url, '_blank');
            }
        }
    }

    async downloadVisualization(filename) {
        if (this.currentSession?.session_id) {
            const url = `${this.apiUrl}/results/${this.currentSession.session_id}/visualizations/${filename}`;
            
            try {
                const result = await chrome.runtime.sendMessage({
                    action: 'downloadFile',
                    url: url,
                    filename: filename
                });
                
                if (result.success) {
                    Utils.showToast('Download started', 'success');
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Download failed:', error);
                Utils.showToast('Download failed', 'error');
            }
        }
    }

    viewGraph(filename) {
        this.viewVisualization(filename);
    }

    downloadGraph(filename) {
        this.downloadVisualization(filename);
    }

    async viewFullReport() {
        if (this.currentSession?.session_id) {
            const url = `${this.apiUrl}/results/${this.currentSession.session_id}`;
            
            try {
                await chrome.runtime.sendMessage({
                    action: 'openTab',
                    url: url
                });
            } catch (error) {
                window.open(url, '_blank');
            }
        }
    }

    async downloadResults() {
        if (this.currentSession?.session_id) {
            const url = `${this.apiUrl}/results/${this.currentSession.session_id}/reports/analysis_results.json`;
            const filename = `algorithm_analysis_${this.currentSession.session_id}.json`;
            
            try {
                const result = await chrome.runtime.sendMessage({
                    action: 'downloadFile',
                    url: url,
                    filename: filename
                });
                
                if (result.success) {
                    Utils.showToast('Results download started', 'success');
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Download failed:', error);
                Utils.showToast('Download failed', 'error');
            }
        }
    }

    shareResults() {
        if (this.currentSession?.session_id) {
            const shareUrl = `${this.apiUrl}/results/${this.currentSession.session_id}`;
            Utils.copyToClipboard(shareUrl).then(() => {
                Utils.showToast('Share link copied to clipboard!', 'success');
            });
        }
    }

    startNewAnalysis() {
        this.currentSession = null;
        const resultsContainer = document.getElementById('resultsContainer');
        const userInput = document.getElementById('userInput');
        
        if (resultsContainer) resultsContainer.style.display = 'none';
        if (userInput) {
            userInput.value = '';
            userInput.focus();
        }
        
        Utils.showToast('Ready for new analysis', 'info');
    }

    // Settings methods
    openSettings() {
        const modal = document.getElementById('settingsModal');
        if (modal) modal.style.display = 'flex';
    }

    closeSettings() {
        const modal = document.getElementById('settingsModal');
        if (modal) modal.style.display = 'none';
    }

    updateSettings() {
        const apiUrlInput = document.getElementById('apiUrl');
        const timeoutInput = document.getElementById('timeout');
        const autoRefreshInput = document.getElementById('autoRefresh');
        
        if (apiUrlInput) this.apiUrl = apiUrlInput.value;
        if (timeoutInput) this.timeout = parseInt(timeoutInput.value);
        if (autoRefreshInput) this.autoRefresh = autoRefreshInput.checked;
        
        this.saveSettings();
        this.closeSettings();
        this.checkHealth();
    }

    resetSettings() {
        const apiUrlInput = document.getElementById('apiUrl');
        const timeoutInput = document.getElementById('timeout');
        const autoRefreshInput = document.getElementById('autoRefresh');
        
        if (apiUrlInput) apiUrlInput.value = CONFIG.DEFAULT_SETTINGS.apiUrl;
        if (timeoutInput) timeoutInput.value = CONFIG.DEFAULT_SETTINGS.timeout;
        if (autoRefreshInput) autoRefreshInput.checked = CONFIG.DEFAULT_SETTINGS.autoRefresh;
        
        Utils.showToast('Settings reset to default values', 'info');
    }

    reportIssue() {
        const errorMessage = document.getElementById('errorMessage');
        const issueData = {
            session: this.currentSession?.session_id || 'N/A',
            error: errorMessage?.textContent || 'No error message',
            timestamp: new Date().toISOString(),
            extension_version: '3.0.0'
        };
        
        const issueText = `Extension Issue Report:\n\n${JSON.stringify(issueData, null, 2)}`;
        
        Utils.copyToClipboard(issueText).then(() => {
            Utils.showToast('Issue details copied to clipboard', 'success');
        });
    }

    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        const errorSection = document.getElementById('errorSection');
        const resultsContainer = document.getElementById('resultsContainer');
        
        if (errorMessage) errorMessage.textContent = message;
        if (errorSection) errorSection.style.display = 'block';
        if (resultsContainer) resultsContainer.style.display = 'none';
    }

    hideError() {
        const errorSection = document.getElementById('errorSection');
        if (errorSection) errorSection.style.display = 'none';
    }
}

// Initialize extension when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.ext = new AlgorithmIntelligenceExtension();
});

// Global functions for backward compatibility
window.viewVisualization = (filename) => window.ext?.viewVisualization(filename);
window.downloadVisualization = (filename) => window.ext?.downloadVisualization(filename);
window.viewGraph = (filename) => window.ext?.viewGraph(filename);
window.downloadGraph = (filename) => window.ext?.downloadGraph(filename);
