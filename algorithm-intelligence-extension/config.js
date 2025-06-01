// config.js - Updated for new app.py
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000',
    API_ENDPOINTS: {
        STATUS: '/api/status',
        ANALYZE: '/api/analyze',
        COMPLEXITY: '/api/complexity',
        SOLVE: '/api/solve',
        VISUALIZE: '/api/visualize',
        ALGORITHMS: '/api/algorithms',
        PERFORMANCE: '/api/performance'
    },
    TIMEOUTS: {
        HEALTH_CHECK: 5000,
        API_REQUEST: 60000
    },
    STORAGE_KEYS: {
        API_URL: 'ais_api_url',
        TIMEOUT: 'ais_timeout',
        AUTO_REFRESH: 'ais_auto_refresh'
    },
    DEFAULT_SETTINGS: {
        apiUrl: 'http://localhost:5000',
        timeout: 60,
        autoRefresh: true
    },
    THEMES: {
        LIGHT: 'light',
        DARK: 'dark',
        AUTO: 'auto'
    },
    ANIMATION_DURATION: 300,
    CACHE_DURATION: 24 * 60 * 60 * 1000, // 24 hours
    MAX_ANALYTICS_EVENTS: 100,
    SUPPORTED_FILE_TYPES: ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.gif'],
    ALGORITHM_CATEGORIES: [
        'searching', 'sorting', 'graphs', 'dynamic_programming', 
        'trees', 'strings', 'linked_lists', 'stacks_queues', 
        'math_number_theory', 'recursion_backtracking', 'greedy', 
        'bit_manipulation', 'sliding_window', 'heap', 'hashing', 'union_find'
    ]
};

// Enhanced utility functions
const Utils = {
    // Time formatting
    formatTime: (seconds) => {
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    },

    // File size formatting
    formatBytes: (bytes) => {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
    },

    // Generate unique ID
    generateId: () => {
        return Math.random().toString(36).substr(2, 9);
    },

    // Enhanced clipboard functionality
    copyToClipboard: async (text) => {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            } else {
                // Fallback for older browsers or non-secure contexts
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                const result = document.execCommand('copy');
                document.body.removeChild(textArea);
                return result;
            }
        } catch (err) {
            console.error('Copy failed:', err);
            return false;
        }
    },

    // Enhanced toast notification system
    showToast: (message, type = 'info', duration = 3000) => {
        // Remove existing toasts of the same type
        document.querySelectorAll(`.toast-${type}`).forEach(toast => toast.remove());
        
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        
        const colors = {
            success: '#4ecdc4',
            error: '#ff6b6b',
            warning: '#fce38a',
            info: '#74b9ff'
        };
        
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${icons[type] || icons.info}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.closest('.toast').remove()">✕</button>
            </div>
        `;
        
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 16px;
            background: ${colors[type] || colors.info};
            color: white;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            z-index: 10000;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            animation: slideInRight 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            max-width: 300px;
            word-wrap: break-word;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    },

    // Date formatting
    formatDate: (date) => {
        if (!date) return 'Unknown';
        const d = new Date(date);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
    },

    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function
    throttle: (func, limit) => {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Local storage helpers
    storage: {
        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('Storage set failed:', e);
                return false;
            }
        },
        
        get: (key, defaultValue = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('Storage get failed:', e);
                return defaultValue;
            }
        },
        
        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('Storage remove failed:', e);
                return false;
            }
        },
        
        clear: () => {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('Storage clear failed:', e);
                return false;
            }
        }
    },

    // URL validation
    isValidUrl: (string) => {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    },

    // File type validation
    isValidFileType: (filename) => {
        const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
        return CONFIG.SUPPORTED_FILE_TYPES.includes(extension);
    },

    // Text processing
    truncateText: (text, maxLength = 100) => {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    },

    // Sanitize HTML
    sanitizeHtml: (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Deep clone object
    deepClone: (obj) => {
        try {
            return JSON.parse(JSON.stringify(obj));
        } catch (e) {
            console.error('Deep clone failed:', e);
            return obj;
        }
    },

    // Check if online
    isOnline: () => {
        return navigator.onLine;
    },

    // Performance measurement
    performance: {
        mark: (name) => {
            if (performance.mark) {
                performance.mark(name);
            }
        },
        
        measure: (name, startMark, endMark) => {
            if (performance.measure) {
                try {
                    performance.measure(name, startMark, endMark);
                    const measure = performance.getEntriesByName(name)[0];
                    return measure ? measure.duration : 0;
                } catch (e) {
                    console.error('Performance measure failed:', e);
                    return 0;
                }
            }
            return 0;
        }
    }
};

// Enhanced CSS for toast animations and additional styles
const additionalStyles = document.createElement('style');
additionalStyles.textContent = `
    /* Toast animations */
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    /* Toast content styling */
    .toast-content {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .toast-icon {
        font-size: 14px;
        flex-shrink: 0;
    }
    
    .toast-message {
        flex: 1;
        line-height: 1.4;
    }
    
    .toast-close {
        background: none;
        border: none;
        color: rgba(255, 255, 255, 0.8);
        cursor: pointer;
        font-size: 14px;
        padding: 0;
        margin-left: 4px;
        flex-shrink: 0;
        transition: color 0.2s ease;
    }
    
    .toast-close:hover {
        color: white;
    }
    
    /* Dark theme support */
    .dark-theme {
        --clay-bg-primary: #1a1a1a;
        --clay-bg-secondary: #2d2d2d;
        --clay-bg-tertiary: #404040;
        --clay-text-primary: #ffffff;
        --clay-text-secondary: #cccccc;
        --clay-text-muted: #999999;
        --clay-shadow-light: rgba(255, 255, 255, 0.1);
        --clay-shadow-dark: rgba(0, 0, 0, 0.8);
    }
    
    /* Additional utility classes */
    .loading {
        pointer-events: none;
        opacity: 0.7;
    }
    
    .disabled {
        opacity: 0.5;
        pointer-events: none;
    }
    
    .hidden {
        display: none !important;
    }
    
    .visible {
        display: block !important;
    }
    
    .text-truncate {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .text-wrap {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Responsive helpers */
    @media (max-width: 480px) {
        .responsive-hide {
            display: none !important;
        }
    }
    
    /* Focus styles for accessibility */
    .clay-button:focus,
    .clay-input:focus,
    .clay-textarea:focus,
    .clay-select:focus {
        outline: 2px solid var(--clay-primary);
        outline-offset: 2px;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .clay-card {
            border: 2px solid var(--clay-text-primary);
        }
        
        .clay-button {
            border: 2px solid var(--clay-text-inverse);
        }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
`;

document.head.appendChild(additionalStyles);

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CONFIG, Utils };
}
