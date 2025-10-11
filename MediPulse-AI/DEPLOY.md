# 🚀 Deploy MediPulse AI to Vercel

This guide will help you deploy the MediPulse AI Hospital Forecasting System to Vercel's serverless platform.

## Prerequisites

1. **Vercel Account**: Create a free account at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm i -g vercel`
3. **Git Repository**: Push your code to GitHub, GitLab, or Bitbucket

## Quick Deployment

### Option 1: Deploy with Vercel CLI (Recommended)

```powershell
# Navigate to your project directory
cd "C:\Users\subha\OneDrive\Desktop\mumbai\Agentic Ai\MediPulse-AI"

# Login to Vercel
vercel login

# Deploy (first time - will prompt for configuration)
vercel

# Follow the prompts:
# - Link to existing project? No
# - Project name: medipulse-ai
# - Directory: ./
# - Override settings? No
```

### Option 2: Deploy from Git Repository

1. **Push to GitHub:**

   ```powershell
   git init
   git add .
   git commit -m "Initial commit - MediPulse AI"
   git remote add origin https://github.com/yourusername/medipulse-ai.git
   git push -u origin main
   ```

2. **Import on Vercel:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

## Configuration

### Environment Variables

Set these environment variables in your Vercel dashboard:

```bash
# Required
NODE_ENV=production
REACT_APP_API_URL=/api
ML_SERVICE_URL=/api/ml

# Optional - SMS Alerts (Twilio)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_phone_number
```

**To add environment variables:**

1. Go to your project dashboard on Vercel
2. Click "Settings" → "Environment Variables"
3. Add each variable above

### Build Settings

Vercel should auto-detect these settings from `vercel.json`:

- **Framework Preset**: Other
- **Build Command**: `npm run vercel-build`
- **Output Directory**: `frontend/build`
- **Install Command**: `npm install`

## Project Structure for Vercel

```
MediPulse-AI/
├── api/                    # Serverless functions
│   ├── predict.js          # Main prediction API
│   ├── health.js           # Health check endpoint
│   └── ml/                 # Python ML functions
│       ├── predict.py      # ML prediction service
│       ├── health.py       # ML health check
│       └── requirements.txt # Python dependencies
├── frontend/               # React application
│   ├── src/
│   ├── public/
│   └── package.json
├── vercel.json            # Vercel configuration
└── package.json           # Root package.json
```

## API Endpoints

After deployment, your API will be available at:

- **Main App**: `https://your-project.vercel.app`
- **Prediction API**: `https://your-project.vercel.app/api/predict`
- **Health Check**: `https://your-project.vercel.app/api/health`
- **ML Prediction**: `https://your-project.vercel.app/api/ml/predict`
- **ML Health**: `https://your-project.vercel.app/api/ml/health`

## Testing Your Deployment

### 1. Health Checks

```powershell
# Test API health
curl https://your-project.vercel.app/api/health

# Test ML health
curl https://your-project.vercel.app/api/ml/health
```

### 2. Make a Prediction

```powershell
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-12-20",
    "day_of_week": 5,
    "month": 12,
    "season": "winter",
    "weather_conditions": "clear",
    "holiday": false,
    "local_events": "none",
    "air_quality_index": 75,
    "city": "Mumbai"
  }'
```

## Custom Domain (Optional)

1. **Add Domain in Vercel:**

   - Go to Project Settings → Domains
   - Add your custom domain (e.g., `medipulse.yourdomain.com`)

2. **Configure DNS:**
   - Add CNAME record pointing to `cname.vercel-dns.com`
   - Or add A record pointing to Vercel's IP

## Monitoring & Analytics

### Built-in Analytics

- Enable Vercel Analytics in Project Settings
- View performance metrics and usage statistics

### Logging

- View function logs in Vercel dashboard
- Use `console.log()` statements for debugging

### Error Monitoring

```javascript
// Add to your API functions for better error tracking
console.error("Error details:", error);
```

## Performance Optimization

### Function Regions

```json
// In vercel.json, specify regions
{
  "functions": {
    "api/predict.js": {
      "maxDuration": 30,
      "regions": ["bom1", "sin1"] // Mumbai, Singapore
    }
  }
}
```

### Caching

- Static assets are cached automatically
- API responses can use cache headers:

```javascript
res.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate");
```

## Troubleshooting

### Common Issues

1. **Build Failures:**

   ```powershell
   # Check build logs in Vercel dashboard
   # Ensure all dependencies are in package.json
   ```

2. **Function Timeouts:**

   ```json
   // Increase timeout in vercel.json
   {
     "functions": {
       "api/predict.js": {
         "maxDuration": 30
       }
     }
   }
   ```

3. **CORS Errors:**

   ```javascript
   // Already handled in API functions
   res.setHeader("Access-Control-Allow-Origin", "*");
   ```

4. **Python Dependencies:**
   ```bash
   # Ensure requirements.txt is in api/ml/
   # Use only supported packages
   ```

### Debugging

1. **View Function Logs:**

   - Vercel Dashboard → Functions → View Logs

2. **Local Development:**

   ```powershell
   # Run Vercel dev server
   vercel dev

   # Access local version
   # http://localhost:3000
   ```

3. **Check Function Status:**
   ```powershell
   vercel ls                    # List deployments
   vercel logs [deployment-url] # View logs
   ```

## Updating Your Deployment

### Automatic Deployments

- Push to main branch triggers auto-deployment
- Preview deployments for pull requests

### Manual Deployments

```powershell
# Redeploy current code
vercel --prod

# Deploy specific branch
vercel --prod --git-branch feature-branch
```

## Cost Optimization

### Vercel Free Tier Limits

- **Function Executions**: 100GB-hours/month
- **Bandwidth**: 100GB/month
- **Build Time**: 6,000 minutes/month

### Optimization Tips

1. **Cache Predictions**: Reduce ML function calls
2. **Optimize Bundle**: Remove unused dependencies
3. **Use Edge Functions**: For faster response times
4. **Monitor Usage**: Track function execution time

## Security Best Practices

1. **Environment Variables**: Store sensitive data in Vercel settings
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Implement in API functions
4. **Input Validation**: Validate all user inputs

## Support

- **Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
- **Community**: [github.com/vercel/community](https://github.com/vercel/community)
- **Project Issues**: Create GitHub issues in your repository

---

Your MediPulse AI application is now ready for global deployment with Vercel's edge network! 🌍
