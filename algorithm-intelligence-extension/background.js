// Algorithm Intelligence Suite - Service Worker Compatible Background Script
class BackgroundService {
    constructor() {
        this.init();
    }

    init() {
        this.setupInstallationHandler();
        this.setupMessageHandlers();
        this.setupContextMenus();
        
        console.log('Algorithm Intelligence Suite background service initialized');
    }

    setupInstallationHandler() {
        chrome.runtime.onInstalled.addListener((details) => {
            console.log('Extension installed/updated:', details);
            
            // Set default settings
            chrome.storage.sync.set({
                'ais_api_url': 'http://localhost:5000/api',
                'ais_timeout': 60,
                'ais_auto_refresh': true,
                'ais_theme': 'auto',
                'ais_user_preferences': {
                    showWelcome: true,
                    enableNotifications: true,
                    enableAnalytics: true
                }
            });

            // Create context menus
            this.createContextMenus();

            // Show welcome notification on first install
            if (details.reason === 'install') {
                this.showWelcomeNotification();
            }
        });
    }

    setupMessageHandlers() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            console.log('Background received message:', request);

            switch (request.action) {
                case 'getHealth':
                    this.checkAPIHealth(request.apiUrl || 'http://localhost:5000')
                        .then(result => sendResponse(result))
                        .catch(error => sendResponse({ success: false, error: error.message }));
                    return true; // Indicates async response

                case 'updateBadge':
                    this.updateBadge(request.text, request.color);
                    sendResponse({ success: true });
                    break;

                case 'downloadFile':
                    this.downloadFile(request.url, request.filename)
                        .then(result => sendResponse(result))
                        .catch(error => sendResponse({ success: false, error: error.message }));
                    return true;

                case 'openTab':
                    chrome.tabs.create({ url: request.url });
                    sendResponse({ success: true });
                    break;

                default:
                    sendResponse({ error: 'Unknown action' });
            }
        });
    }

    createContextMenus() {
        // Remove all existing context menus first
        chrome.contextMenus.removeAll(() => {
            // Add context menu for selected text
            chrome.contextMenus.create({
                id: 'analyzeSelection',
                title: 'Analyze with Algorithm Intelligence',
                contexts: ['selection']
            });

            // Add context menu for page scanning
            chrome.contextMenus.create({
                id: 'scanPage',
                title: 'Scan Page for Algorithms',
                contexts: ['page']
            });
        });
    }

    setupContextMenus() {
        chrome.contextMenus.onClicked.addListener((info, tab) => {
            this.handleContextMenuClick(info, tab);
        });
    }

    handleContextMenuClick(info, tab) {
        switch (info.menuItemId) {
            case 'analyzeSelection':
                // Store selected text for popup to use
                chrome.storage.local.set({
                    'selectedText': info.selectionText,
                    'selectionTimestamp': Date.now()
                });
                // Open popup
                chrome.action.openPopup();
                break;

            case 'scanPage':
                // Open popup and trigger scan
                chrome.storage.local.set({
                    'triggerScan': true,
                    'scanTimestamp': Date.now()
                });
                chrome.action.openPopup();
                break;
        }
    }

    async checkAPIHealth(apiUrl) {
        try {
            const response = await fetch(`${apiUrl}/api/status`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(5000)
            });

            if (response.ok) {
                const data = await response.json();
                this.updateBadge('✓', '#4CAF50');
                return { 
                    success: true, 
                    status: 'online',
                    data: data
                };
            } else {
                this.updateBadge('!', '#FF9800');
                return { 
                    success: false, 
                    status: 'error',
                    error: `HTTP ${response.status}`
                };
            }

        } catch (error) {
            this.updateBadge('✗', '#F44336');
            return { 
                success: false, 
                status: 'offline',
                error: error.message
            };
        }
    }

    updateBadge(text, color = '#4CAF50') {
        chrome.action.setBadgeText({ text: text || '' });
        chrome.action.setBadgeBackgroundColor({ color: color });
    }

    async downloadFile(url, filename) {
        try {
            const downloadId = await chrome.downloads.download({
                url: url,
                filename: filename || 'algorithm_file',
                saveAs: false
            });

            return { success: true, downloadId: downloadId };
        } catch (error) {
            console.error('Download failed:', error);
            return { success: false, error: error.message };
        }
    }

    showWelcomeNotification() {
        if (chrome.notifications) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: 'Algorithm Intelligence Suite',
                message: 'Extension installed successfully! Click the icon to get started.'
            });
        }
    }
}

// Initialize background service
new BackgroundService();

// Handle extension startup
chrome.runtime.onStartup.addListener(() => {
    console.log('Extension startup detected');
});
