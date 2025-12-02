// Cookie Consent Management for GDPR Compliance
class CookieConsent {
  constructor() {
    this.consentKey = 'cookie-consent';
    this.consentGiven = this.getConsent();
    this.categories = {
      necessary: { required: true, description: 'Essential cookies for basic site functionality' },
      analytics: { required: false, description: 'Help us improve our website by collecting anonymous usage data' },
      marketing: { required: false, description: 'Used to deliver personalized advertisements' }
    };

    this.init();
  }

  init() {
    if (!this.consentGiven) {
      this.showBanner();
    } else {
      this.applyConsent();
    }
  }

  getConsent() {
    try {
      const consent = localStorage.getItem(this.consentKey);
      return consent ? JSON.parse(consent) : null;
    } catch {
      return null;
    }
  }

  saveConsent(consent) {
    try {
      localStorage.setItem(this.consentKey, JSON.stringify(consent));
      this.consentGiven = consent;
      this.applyConsent();
      this.hideBanner();
    } catch (error) {
      console.warn('Failed to save cookie consent:', error);
    }
  }

  applyConsent() {
    if (!this.consentGiven) return;

    // Apply analytics consent
    if (this.consentGiven.analytics) {
      this.enableAnalytics();
    } else {
      this.disableAnalytics();
    }

    // Apply marketing consent
    if (this.consentGiven.marketing) {
      this.enableMarketing();
    } else {
      this.disableMarketing();
    }
  }

  enableAnalytics() {
    // Enable Google Analytics
    if (typeof gtag === 'function') {
      gtag('consent', 'update', {
        'analytics_storage': 'granted',
        'ad_storage': 'denied'
      });
    }

    // Enable Cloudflare Analytics
    const cfScript = document.querySelector('script[data-cf-beacon]');
    if (cfScript) {
      cfScript.removeAttribute('data-cf-beacon-blocked');
    }

    // Enable custom analytics
    window.analyticsConsent = true;
  }

  disableAnalytics() {
    // Disable Google Analytics
    if (typeof gtag === 'function') {
      gtag('consent', 'update', {
        'analytics_storage': 'denied',
        'ad_storage': 'denied'
      });
    }

    // Disable Cloudflare Analytics
    const cfScript = document.querySelector('script[data-cf-beacon]');
    if (cfScript) {
      cfScript.setAttribute('data-cf-beacon-blocked', 'true');
    }

    // Disable custom analytics
    window.analyticsConsent = false;
  }

  enableMarketing() {
    // Enable marketing cookies
    window.marketingConsent = true;
  }

  disableMarketing() {
    // Disable marketing cookies
    window.marketingConsent = false;
  }

  showBanner() {
    const banner = document.createElement('div');
    banner.id = 'cookie-consent-banner';
    banner.innerHTML = `
      <div class="cookie-consent-overlay">
        <div class="cookie-consent-modal">
          <div class="cookie-consent-header">
            <h3>🍪 Privacy & Cookies</h3>
            <p>We use cookies to enhance your experience and analyze site traffic. Your privacy matters.</p>
          </div>

          <div class="cookie-consent-categories">
            ${Object.entries(this.categories).map(([key, config]) => `
              <div class="cookie-category">
                <label class="cookie-toggle">
                  <input type="checkbox"
                         id="cookie-${key}"
                         ${config.required ? 'checked disabled' : ''}
                         ${!config.required && this.consentGiven?.[key] ? 'checked' : ''}>
                  <span class="cookie-slider ${config.required ? 'disabled' : ''}"></span>
                </label>
                <div class="cookie-info">
                  <strong>${key.charAt(0).toUpperCase() + key.slice(1)}</strong>
                  <span>${config.description}</span>
                  ${config.required ? '<small>(Required)</small>' : ''}
                </div>
              </div>
            `).join('')}
          </div>

          <div class="cookie-consent-actions">
            <button class="cookie-btn cookie-btn-secondary" onclick="cookieConsent.acceptSelected()">
              Accept Selected
            </button>
            <button class="cookie-btn cookie-btn-primary" onclick="cookieConsent.acceptAll()">
              Accept All
            </button>
          </div>

          <div class="cookie-consent-footer">
            <a href="#privacy-policy" onclick="cookieConsent.showPrivacyPolicy()">Privacy Policy</a>
            <a href="#cookie-details" onclick="cookieConsent.showDetails()">Cookie Details</a>
          </div>
        </div>
      </div>
    `;

    // Add styles
    const style = document.createElement('style');
    style.textContent = `
      .cookie-consent-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
      }

      .cookie-consent-modal {
        background: var(--bg-card, #0a0a0a);
        border: 1px solid var(--border, #1f2937);
        border-radius: 12px;
        padding: 2rem;
        max-width: 500px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
      }

      .cookie-consent-header h3 {
        color: var(--text-main, #ffffff);
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
      }

      .cookie-consent-header p {
        color: var(--text-muted, #9ca3af);
        margin: 0 0 1.5rem 0;
        line-height: 1.5;
      }

      .cookie-consent-categories {
        margin-bottom: 1.5rem;
      }

      .cookie-category {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
      }

      .cookie-toggle {
        position: relative;
        display: inline-block;
        width: 48px;
        height: 24px;
        flex-shrink: 0;
        margin-top: 2px;
      }

      .cookie-toggle input {
        opacity: 0;
        width: 0;
        height: 0;
      }

      .cookie-slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: var(--border, #1f2937);
        transition: 0.3s;
        border-radius: 24px;
      }

      .cookie-slider:before {
        position: absolute;
        content: "";
        height: 18px;
        width: 18px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: 0.3s;
        border-radius: 50%;
      }

      .cookie-toggle input:checked + .cookie-slider {
        background-color: var(--accent, #00ff41);
      }

      .cookie-toggle input:checked + .cookie-slider:before {
        transform: translateX(24px);
      }

      .cookie-slider.disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }

      .cookie-info {
        flex: 1;
      }

      .cookie-info strong {
        color: var(--text-main, #ffffff);
        display: block;
        margin-bottom: 0.25rem;
      }

      .cookie-info span {
        color: var(--text-muted, #9ca3af);
        font-size: 0.875rem;
        line-height: 1.4;
      }

      .cookie-info small {
        color: var(--accent, #00ff41);
        font-size: 0.75rem;
      }

      .cookie-consent-actions {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
      }

      .cookie-btn {
        flex: 1;
        padding: 0.75rem 1.5rem;
        border: 1px solid var(--border, #1f2937);
        border-radius: 6px;
        font-family: inherit;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      }

      .cookie-btn-primary {
        background: var(--accent, #00ff41);
        color: var(--bg-main, #050505);
        border-color: var(--accent, #00ff41);
      }

      .cookie-btn-primary:hover {
        opacity: 0.8;
      }

      .cookie-btn-secondary {
        background: transparent;
        color: var(--text-main, #ffffff);
      }

      .cookie-btn-secondary:hover {
        background: rgba(255, 255, 255, 0.1);
      }

      .cookie-consent-footer {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
      }

      .cookie-consent-footer a {
        color: var(--accent, #00ff41);
        text-decoration: none;
      }

      .cookie-consent-footer a:hover {
        text-decoration: underline;
      }

      @media (max-width: 640px) {
        .cookie-consent-modal {
          padding: 1.5rem;
        }

        .cookie-category {
          flex-direction: column;
          gap: 0.5rem;
        }

        .cookie-consent-actions {
          flex-direction: column;
        }

        .cookie-consent-footer {
          flex-direction: column;
          gap: 0.5rem;
          text-align: center;
        }
      }
    `;

    document.head.appendChild(style);
    document.body.appendChild(banner);

    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
  }

  hideBanner() {
    const banner = document.getElementById('cookie-consent-banner');
    if (banner) {
      banner.remove();
      document.body.style.overflow = '';
    }
  }

  acceptSelected() {
    const consent = {
      necessary: true,
      analytics: document.getElementById('cookie-analytics')?.checked || false,
      marketing: document.getElementById('cookie-marketing')?.checked || false
    };
    this.saveConsent(consent);
  }

  acceptAll() {
    const consent = {
      necessary: true,
      analytics: true,
      marketing: true
    };
    this.saveConsent(consent);
  }

  showPrivacyPolicy() {
    // Show privacy policy details
    alert('Privacy Policy: We collect anonymous analytics data to improve our website. No personal information is stored. You can withdraw consent at any time.');
  }

  showDetails() {
    // Show detailed cookie information
    alert('Cookie Details: \n\n• Necessary: Essential for site functionality\n• Analytics: Anonymous usage statistics\n• Marketing: Personalized content\n\nYou can change preferences anytime in your browser settings.');
  }

  withdraw() {
    localStorage.removeItem(this.consentKey);
    this.consentGiven = null;
    this.disableAnalytics();
    this.disableMarketing();
    this.showBanner();
  }
}

// Initialize cookie consent when DOM is ready
let cookieConsent;
document.addEventListener('DOMContentLoaded', function() {
  cookieConsent = new CookieConsent();

  // Add consent management button to footer
  const footer = document.querySelector('footer');
  if (footer) {
    const manageBtn = document.createElement('button');
    manageBtn.textContent = '🍪 Manage Cookies';
    manageBtn.style.cssText = `
      background: transparent;
      border: 1px solid var(--border, #1f2937);
      color: var(--text-muted, #9ca3af);
      padding: 0.5rem 1rem;
      border-radius: 4px;
      font-size: 0.75rem;
      cursor: pointer;
      margin-top: 1rem;
      transition: all 0.2s;
    `;
    manageBtn.onmouseover = () => manageBtn.style.color = 'var(--accent, #00ff41)';
    manageBtn.onmouseout = () => manageBtn.style.color = 'var(--text-muted, #9ca3af)';
    manageBtn.onclick = () => cookieConsent.withdraw();
    footer.appendChild(manageBtn);
  }
});
