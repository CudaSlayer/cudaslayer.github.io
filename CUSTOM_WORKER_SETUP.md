# Custom Analytics Worker Setup Guide

## 🤔 What is the Custom Worker For?

The **Custom Analytics Worker** is your own private analytics system that gives you complete control over your data. Here's what it does:

### Purpose & Benefits
- **Self-hosted analytics**: No dependence on third parties for core tracking
- **Complete data ownership**: All data stored in your GitHub repository
- **Advanced custom tracking**: Track specific user interactions that GA/Cloudflare can't
- **Cross-site analytics**: Combine data from both portfolio and blog sites
- **Cost-effective**: Free tier handles most use cases
- **Privacy compliant**: You control exactly what's stored and for how long

### What It Tracks
1. **Homepage visits** - From your portfolio page
2. **Custom events** - User interactions like theme changes, link clicks
3. **Analytics aggregation** - Combines data from both sites
4. **Data persistence** - Stores data securely in GitHub

### Why Use It Alongside GA/Cloudflare?
- **Redundancy**: If one service fails, you still have data
- **Comparison**: Validate data between systems
- **Customization**: Track things GA can't (e.g., specific button clicks)
- **Control**: Full ownership of raw data
- **Learning**: Understand how analytics systems work

## 🚀 Step-by-Step Setup

### Prerequisites
- GitHub account (you have this)
- Cloudflare account (free tier works)
- Basic command line knowledge
- 15-20 minutes of time

---

## Step 1: Install Wrangler CLI

```bash
# Install Wrangler (Cloudflare's CLI tool)
npm install -g wrangler

# Verify installation
wrangler --version
```

---

## Step 2: Create GitHub Personal Access Token

1. **Go to GitHub Settings**: [github.com/settings/tokens](https://github.com/settings/tokens)
2. **Click "Generate new token" → "Generate new token (classic)**
3. **Configure token**:
   - **Note**: "Analytics Worker Token"
   - **Expiration**: 90 days (or No expiration)
   - **Scopes**: Check only `repo` (Full control of private repositories)
4. **Click "Generate token"**
5. **Copy the token immediately** (you can't see it again!)

---

## Step 3: Create GitHub Comment for Data Storage

The worker stores analytics data in a GitHub issue comment as JSON. Here's how:

1. **Create a new issue** in your repository:
   - Go to: [github.com/CudaSlayer/cudaslayer.github.io/issues/new](https://github.com/CudaSlayer/cudaslayer.github.io/issues/new)
   - Title: "Analytics Data Storage"
   - Description: "This comment stores analytics data for the custom worker."
   - Click "Submit new issue"

2. **Add a comment** to the issue:
   - Comment: `{"slugs": {}}`
   - Click "Comment"

3. **Copy the comment ID** from the URL:
   - URL will be: `github.com/CudaSlayer/cudaslayer.github.io/issues/1#issuecomment-1234567890`
   - The comment ID is: `1234567890`

---

## Step 4: Configure Cloudflare Worker

### 4.1 Login to Cloudflare
```bash
wrangler login
# This will open a browser to authenticate
```

### 4.2 Create Worker Project
```bash
# Navigate to your project directory
cd /path/to/your/project

# Create worker
wrangler init analytics-worker
# Choose: "Hello World" example
# Choose: TypeScript (or JavaScript)
# Choose: No for deployment
```

### 4.3 Replace Worker Code
```bash
# Copy your enhanced worker code
cp workers/github-metrics.js analytics-worker/src/index.js
```

### 4.4 Configure Worker Environment Variables

```bash
# Set each environment variable
wrangler secret put GITHUB_OWNER
# When prompted, enter: CudaSlayer

wrangler secret put GITHUB_REPO
# When prompted, enter: cudaslayer.github.io

wrangler secret put GITHUB_TOKEN
# When prompted, paste your GitHub token

wrangler secret put GITHUB_COMMENT_ID
# When prompted, enter your comment ID (1234567890)
```

### 4.5 Deploy Worker
```bash
# Deploy to Cloudflare
wrangler deploy

# You'll get a URL like: https://analytics-worker.your-subdomain.workers.dev
```

---

## Step 5: Update Your HTML Files

Now update the placeholder URLs in your HTML files:

### In `index.html` and `blog.html`:
```javascript
// Find this line (around line 1130 in index.html):
const METRICS_BASE_URL = 'https://your-worker-subdomain.workers.dev';

// Replace with your actual worker URL:
const METRICS_BASE_URL = 'https://analytics-worker.your-subdomain.workers.dev';
```

---

## 🔧 Configuration Summary

### Environment Variables Set:
```bash
GITHUB_OWNER=CudaSlayer
GITHUB_REPO=cudaslayer.github.io
GITHUB_TOKEN=ghp_your_personal_access_token
GITHUB_COMMENT_ID=1234567890
```

### HTML Files Updated:
- Cloudflare token replaced
- Google Analytics ID replaced
- Custom worker URL updated

---

## 🧪 Testing Your Worker

### Test Direct API Calls:
```bash
# Test analytics endpoint
curl -X GET https://your-worker-url.workers.dev/analytics-summary

# Test event tracking
curl -X POST https://your-worker-url.workers.dev/event \
  -H "Content-Type: application/json" \
  -d '{"eventType":"test","page":"home","element":"button"}'
```

### Test in Browser:
1. Open your website
2. Open browser DevTools (F12)
3. Go to Network tab
4. Look for requests to your worker URL
5. Check Console for any errors

---

## 📊 What Data Gets Stored

### In GitHub Comment:
```json
{
  "slugs": {
    "homepage": {
      "views": 15,
      "lastVisit": "2025-12-02T20:00:00.000Z",
      "visits": [
        {
          "timestamp": "2025-12-02T20:00:00.000Z",
          "userAgent": "Mozilla/5.0...",
          "referrer": "https://google.com"
        }
      ]
    },
    "cuda-day1": {
      "views": 8,
      "likes": 2
    },
    "events": [
      {
        "eventType": "theme_change",
        "page": "portfolio",
        "element": "sepia_theme",
        "timestamp": "2025-12-02T20:00:00.000Z"
      }
    ]
  }
}
```

---

## 🔍 Troubleshooting

### Common Issues:

**Worker returns 404 error:**
```bash
# Check if worker deployed correctly
wrangler whoami
wrangler deploy
```

**Permission denied from GitHub:**
```bash
# Verify token has repo scope
# Check GITHUB_OWNER and GITHUB_REPO are correct
# Ensure token isn't expired
```

**No data appearing:**
```bash
# Check environment variables
wrangler secret list

# Test API directly
curl https://your-worker-url.workers.dev/analytics-summary
```

**CORS errors in browser:**
- Check worker's CORS headers in the code
- Ensure your site domain is allowed
- Test with curl first

---

## 📈 Monitoring Your Worker

### Cloudflare Dashboard:
1. Go to [dash.cloudflare.com](https://dash.cloudflare.com)
2. Navigate to "Workers & Pages"
3. Click on your worker
4. View "Analytics" tab for usage stats
5. Check "Logs" for debugging

### GitHub Storage:
1. Go to your analytics issue
2. Check the comment for updated data
3. View raw JSON structure

---

## 🔄 Maintenance

### Regular Tasks:
- **Weekly**: Check worker usage (stays under free tier)
- **Monthly**: Verify GitHub token hasn't expired
- **Quarterly**: Review and clean old data
- **Yearly**: Update token for security

### Backup Data:
The GitHub comment acts as your database. Consider:
- Monthly JSON exports
- Repository backups
- Data analysis scripts

---

## 🎯 Success Criteria

Your worker is working when:
- ✅ `curl` commands return valid responses
- ✅ Browser network tab shows worker requests
- ✅ GitHub comment updates with visit data
- ✅ Analytics summary endpoint shows aggregated data
- ✅ No CORS errors in browser console
- ✅ Worker stays within free tier limits

---

## 💡 Advanced Options

### Custom Analytics Dashboard:
- Create a simple React/Vue dashboard
- Fetch data from your worker's API
- Visualize visitor patterns and trends

### Data Processing:
- Set up GitHub Actions for data analysis
- Create automated reports
- Export to CSV/Google Sheets

### Enhanced Security:
- Use GitHub Actions for token rotation
- Implement request rate limiting
- Add authentication for admin endpoints

---

## 📞 Need Help?

### Resources:
- **Cloudflare Workers Docs**: [developers.cloudflare.com/workers](https://developers.cloudflare.com/workers)
- **Wrangler CLI**: [developers.cloudflare.com/workers/wrangler](https://developers.cloudflare.com/workers/wrangler)
- **GitHub API**: [docs.github.com/en/rest](https://docs.github.com/en/rest)

### Quick Commands Reference:
```bash
# Deploy worker
wrangler deploy

# Check logs
wrangler tail

# List secrets
wrangler secret list

# Update worker code
wrangler deploy

# Delete worker
wrangler delete analytics-worker
```

---

## 🎉 You're Done!

Once deployed, your custom analytics worker will:
- Track homepage visits from your portfolio
- Record custom events (theme changes, link clicks)
- Store data securely in your GitHub repository
- Provide analytics summaries via API
- Work alongside GA and Cloudflare Analytics

You now have **complete control** over your analytics data while maintaining professional-grade visitor tracking capabilities!
