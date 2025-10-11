const express = require('express');
const axios = require('axios');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const winston = require('winston');
const twilio = require('twilio');
require('dotenv').config();

// Setup logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

const app = express();
const PORT = process.env.PORT || 3001;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Twilio setup
let twilioClient = null;
if (process.env.TWILIO_ACCOUNT_SID && 
    process.env.TWILIO_AUTH_TOKEN && 
    process.env.TWILIO_ACCOUNT_SID.startsWith('AC')) {
  try {
    twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
    logger.info('Twilio client initialized');
  } catch (error) {
    logger.warn('Failed to initialize Twilio client:', error.message);
    twilioClient = null;
  }
} else {
  logger.warn('Twilio credentials not provided or invalid - SMS alerts disabled');
}

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));

// Create logs directory
const fs = require('fs');
if (!fs.existsSync('logs')) {
  fs.mkdirSync('logs');
}

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: { error: 'Too many requests from this IP' }
});
app.use('/api', limiter);

// Request logging middleware
app.use((req, res, next) => {
  req.startTime = Date.now();
  logger.info({
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });
  next();
});

// Resource Calculator Class
class ResourceCalculator {
  constructor() {
    this.config = this.getDefaultConfig();
  }

  getDefaultConfig() {
    return {
      staffing: {
        doctors: {
          patientsPerDoctor: 20,
          minimumCount: 2,
          overtimeThreshold: 150,
          departments: {
            emergency: { ratio: 0.3, minCount: 1 },
            pulmonology: { ratio: 0.2, minCount: 1 },
            general: { ratio: 0.5, minCount: 1 }
          }
        },
        nurses: {
          patientsPerNurse: 8,
          minimumCount: 5,
          shiftMultiplier: 3
        }
      },
      supplies: {
        oxygenCylinders: {
          patientsPerUnit: 10,
          bufferPercentage: 0.2,
          minimumStock: 5
        },
        ventilators: {
          patientsPerUnit: 100,
          criticalCareRatio: 0.1,
          minimumStock: 2
        },
        beds: {
          occupancyTarget: 0.85,
          bufferPercentage: 0.15
        }
      },
      costs: {
        doctor: { dailyRate: 5000, overtimeMultiplier: 1.5 },
        nurse: { dailyRate: 2000, overtimeMultiplier: 1.5 },
        oxygenCylinder: { unitCost: 500 },
        ventilator: { dailyRental: 2000 },
        bed: { dailyRate: 1000 }
      },
      thresholds: {
        surge: { low: 120, medium: 180, high: 250 }
      }
    };
  }

  calculateStaffing(predictedPatients, context = {}) {
    const { aqi = 100, outbreak = 0 } = context;
    const config = this.config.staffing;

    // Doctor calculation
    let doctorCount = Math.max(
      config.doctors.minimumCount,
      Math.ceil(predictedPatients / config.doctors.patientsPerDoctor)
    );

    // AQI adjustment for pulmonology
    if (aqi > 200) {
      const aqiMultiplier = Math.min(1.5, 1 + (aqi - 200) / 1000);
      doctorCount = Math.ceil(doctorCount * aqiMultiplier);
    }

    // Nurse calculation
    const nurseCount = Math.max(
      config.nurses.minimumCount,
      Math.ceil(predictedPatients / config.nurses.patientsPerNurse)
    ) * config.nurses.shiftMultiplier;

    const isOvertime = predictedPatients > config.doctors.overtimeThreshold;

    return {
      doctors: {
        total: doctorCount,
        overtime: isOvertime,
        cost: this.calculateStaffCost('doctor', doctorCount, isOvertime)
      },
      nurses: {
        total: nurseCount,
        cost: this.calculateStaffCost('nurse', nurseCount, false)
      },
      totalCost: this.calculateStaffCost('doctor', doctorCount, isOvertime) +
                 this.calculateStaffCost('nurse', nurseCount, false)
    };
  }

  calculateSupplies(predictedPatients, context = {}) {
    const { aqi = 100, icuRatio = 0.15 } = context;
    const config = this.config.supplies;

    // Oxygen calculation
    let oxygenNeeded = Math.ceil(predictedPatients / config.oxygenCylinders.patientsPerUnit);
    if (aqi > 150) {
      const aqiMultiplier = 1 + Math.min(0.5, (aqi - 150) / 500);
      oxygenNeeded = Math.ceil(oxygenNeeded * aqiMultiplier);
    }
    const oxygenRecommended = Math.max(
      config.oxygenCylinders.minimumStock,
      Math.ceil(oxygenNeeded * (1 + config.oxygenCylinders.bufferPercentage))
    );

    // Ventilator calculation
    const icuPatients = Math.ceil(predictedPatients * icuRatio);
    const ventilatorNeeded = Math.max(
      config.ventilators.minimumStock,
      Math.ceil(icuPatients * config.ventilators.criticalCareRatio)
    );

    // Bed calculation
    const bedsNeeded = Math.ceil(predictedPatients / config.beds.occupancyTarget);
    const bedsRecommended = Math.ceil(bedsNeeded * (1 + config.beds.bufferPercentage));

    return {
      oxygen: {
        required: oxygenNeeded,
        recommended: oxygenRecommended,
        cost: this.calculateSupplyCost('oxygenCylinder', oxygenRecommended)
      },
      ventilators: {
        required: ventilatorNeeded,
        icuPatients,
        cost: this.calculateSupplyCost('ventilator', ventilatorNeeded, true)
      },
      beds: {
        required: bedsNeeded,
        recommended: bedsRecommended,
        cost: this.calculateSupplyCost('bed', bedsRecommended, true)
      }
    };
  }

  calculateStaffCost(staffType, count, isOvertime = false) {
    const config = this.config.costs[staffType];
    const baseRate = config.dailyRate * count;
    return isOvertime ? baseRate * config.overtimeMultiplier : baseRate;
  }

  calculateSupplyCost(itemType, quantity, isDaily = false) {
    const config = this.config.costs[itemType];
    const rate = isDaily ? config.dailyRental : config.unitCost;
    return rate * quantity;
  }

  generateRecommendations(predictedPatients, context = {}) {
    const staffing = this.calculateStaffing(predictedPatients, context);
    const supplies = this.calculateSupplies(predictedPatients, context);
    
    return {
      staffing,
      supplies,
      totalCost: staffing.totalCost + supplies.oxygen.cost + supplies.ventilators.cost + supplies.beds.cost,
      surgeLevel: this.determineSurgeLevel(predictedPatients),
      actionItems: this.generateActionItems(predictedPatients, context, staffing, supplies)
    };
  }

  determineSurgeLevel(patients) {
    const thresholds = this.config.thresholds.surge;
    if (patients >= thresholds.high) return 'critical';
    if (patients >= thresholds.medium) return 'high';
    if (patients >= thresholds.low) return 'medium';
    return 'normal';
  }

  generateActionItems(predictedPatients, context, staffing, supplies) {
    const actions = [];
    
    if (staffing.doctors.overtime) {
      actions.push({
        priority: 'high',
        category: 'staffing',
        action: 'Activate overtime protocols for medical staff',
        timeline: 'immediate'
      });
    }

    if (context.aqi > 300) {
      actions.push({
        priority: 'high',
        category: 'environmental',
        action: 'Issue public health advisory for respiratory risk groups',
        timeline: 'within 2h'
      });
    }

    if (context.outbreak > 1) {
      actions.push({
        priority: 'critical',
        category: 'infection_control',
        action: 'Activate infection control protocols',
        timeline: 'immediate'
      });
    }

    if (predictedPatients > 200) {
      actions.push({
        priority: 'high',
        category: 'capacity',
        action: 'Prepare surge capacity and alert management',
        timeline: 'within 4h'
      });
    }

    return actions.sort((a, b) => {
      const priorityOrder = { critical: 3, high: 2, medium: 1, low: 0 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }
}

// Initialize resource calculator
const resourceCalculator = new ResourceCalculator();

// Axios instance for FastAPI calls
const apiClient = axios.create({
  baseURL: FASTAPI_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Response interceptor for API calls
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    logger.error('FastAPI request failed:', {
      url: error.config?.url,
      status: error.response?.status,
      message: error.message
    });
    return Promise.reject(error);
  }
);

// Validation middleware
const validatePredictRequest = [
  body('city').notEmpty().isLength({ min: 1, max: 100 }),
  body('date').isISO8601(),
  body('aqi').isInt({ min: 0, max: 500 }),
  body('temperature').isFloat({ min: -50, max: 60 }),
  body('festival').optional().isLength({ max: 50 }),
  body('outbreak').optional().isInt({ min: 0, max: 3 })
];

// Advisory generation function
function generateAdvisories(input, prediction) {
  const advisories = [];
  const { aqi, temperature, outbreak } = input;
  const patients = prediction.predicted_patients;

  // AQI-based advisories
  if (aqi > 300) {
    advisories.push({
      type: 'environmental',
      severity: 'critical',
      title: 'Hazardous Air Quality Alert',
      message: 'Extremely poor air quality detected. Immediate respiratory support preparation required.',
      recommendations: [
        'Increase oxygen cylinder inventory by 30%',
        'Alert all pulmonology staff',
        'Issue public health advisory to high-risk groups',
        'Prepare additional respiratory equipment'
      ]
    });
  } else if (aqi > 200) {
    advisories.push({
      type: 'environmental',
      severity: 'high',
      title: 'Poor Air Quality Warning',
      message: 'Elevated AQI may increase respiratory-related admissions.',
      recommendations: [
        'Monitor respiratory patients closely',
        'Ensure adequate oxygen supply',
        'Brief staff on air quality impacts'
      ]
    });
  }

  // Patient surge advisories
  if (patients > 250) {
    advisories.push({
      type: 'capacity',
      severity: 'critical',
      title: 'Critical Patient Surge Predicted',
      message: 'Extremely high patient volume expected. Immediate surge protocols required.',
      recommendations: [
        'Activate all available staff',
        'Prepare overflow areas immediately',
        'Coordinate with nearby hospitals for potential transfers',
        'Alert hospital administration'
      ]
    });
  } else if (patients > 180) {
    advisories.push({
      type: 'capacity',
      severity: 'high',
      title: 'High Patient Volume Expected',
      message: 'Significant increase in patient admissions anticipated.',
      recommendations: [
        'Schedule additional staff shifts',
        'Prepare additional bed capacity',
        'Review discharge planning for stable patients'
      ]
    });
  }

  // Outbreak advisories
  if (outbreak > 2) {
    advisories.push({
      type: 'outbreak',
      severity: 'critical',
      title: 'Epidemic Level Outbreak',
      message: 'Major outbreak detected. Comprehensive infection control required.',
      recommendations: [
        'Implement strict isolation protocols',
        'Maximize PPE availability',
        'Coordinate with health authorities',
        'Prepare quarantine facilities'
      ]
    });
  } else if (outbreak > 0) {
    advisories.push({
      type: 'outbreak',
      severity: 'high',
      title: 'Active Outbreak Detected',
      message: 'Localized outbreak requires enhanced precautions.',
      recommendations: [
        'Increase infection control measures',
        'Ensure adequate PPE stocks',
        'Monitor patient symptoms closely'
      ]
    });
  }

  return advisories;
}

// Routes
app.get('/', (req, res) => {
  res.json({
    service: 'MediPulse AI Backend',
    version: '1.0.0',
    status: 'healthy',
    endpoints: {
      predict: 'POST /api/predict',
      forecast: 'POST /api/forecast',
      batch: 'POST /api/predict/batch',
      alerts: 'POST /api/alerts/send',
      health: 'GET /api/health'
    },
    timestamp: new Date().toISOString()
  });
});

app.get('/api/health', async (req, res) => {
  try {
    const response = await apiClient.get('/health');
    res.json({
      backend: {
        status: 'healthy',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        timestamp: new Date().toISOString()
      },
      fastapi: response.data,
      services: {
        twilio: twilioClient ? 'available' : 'disabled',
        logging: 'active'
      }
    });
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      backend: {
        status: 'healthy',
        uptime: process.uptime(),
        timestamp: new Date().toISOString()
      },
      fastapi: {
        status: 'unavailable',
        error: error.message
      },
      services: {
        twilio: twilioClient ? 'available' : 'disabled',
        logging: 'active'
      }
    });
  }
});

app.post('/api/predict', validatePredictRequest, async (req, res) => {
  try {
    // Validate request
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array()
      });
    }

    // Call FastAPI service
    const response = await apiClient.post('/predict', req.body);
    const prediction = response.data;

    // Calculate resource recommendations
    const context = {
      aqi: req.body.aqi,
      temperature: req.body.temperature,
      outbreak: req.body.outbreak || 0
    };
    
    const recommendations = resourceCalculator.generateRecommendations(
      prediction.predicted_patients,
      context
    );

    // Generate advisories
    const advisories = generateAdvisories(req.body, prediction);

    // Compose enhanced response
    const result = {
      ...prediction,
      recommendations,
      advisories,
      context: {
        surge_level: recommendations.surgeLevel,
        risk_factors: {
          high_aqi: req.body.aqi > 200,
          extreme_temperature: req.body.temperature < 10 || req.body.temperature > 35,
          active_outbreak: req.body.outbreak > 0,
          festival_impact: !!req.body.festival
        }
      },
      metadata: {
        processing_time: Date.now() - req.startTime,
        request_id: req.get('X-Request-ID') || `req_${Date.now()}`,
        api_version: '1.0'
      }
    };

    logger.info(`Prediction completed for ${req.body.city}: ${prediction.predicted_patients} patients`);
    res.json(result);

  } catch (error) {
    logger.error('Prediction endpoint error:', error);
    
    if (error.response?.status === 422) {
      res.status(400).json({
        error: 'Invalid input data',
        details: error.response.data.detail
      });
    } else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      res.status(503).json({
        error: 'Prediction service unavailable',
        message: 'ML service is not responding. Please try again later.',
        code: 'SERVICE_UNAVAILABLE'
      });
    } else {
      res.status(500).json({
        error: 'Internal server error',
        message: 'An unexpected error occurred during prediction',
        code: 'INTERNAL_ERROR'
      });
    }
  }
});

app.post('/api/forecast', async (req, res) => {
  try {
    const response = await apiClient.post('/forecast', req.body);
    const forecast = response.data;

    // Enhance each day's prediction with recommendations
    const enhancedPredictions = forecast.predictions.map(dayPred => {
      const context = {
        aqi: req.body.current_aqi || 150,
        outbreak: req.body.current_outbreak || 0
      };
      
      const recommendations = resourceCalculator.generateRecommendations(
        dayPred.predicted_patients,
        context
      );

      return {
        ...dayPred,
        surge_level: recommendations.surgeLevel,
        staffing_needed: recommendations.staffing,
        estimated_cost: recommendations.totalCost
      };
    });

    // Calculate summary statistics
    const totalCost = enhancedPredictions.reduce((sum, day) => sum + day.estimated_cost, 0);
    const avgSurgeLevel = enhancedPredictions.reduce((count, day) => {
      return count + (day.surge_level !== 'normal' ? 1 : 0);
    }, 0);

    res.json({
      ...forecast,
      predictions: enhancedPredictions,
      summary: {
        ...forecast.summary,
        total_estimated_cost: totalCost,
        surge_days: avgSurgeLevel,
        recommendations: resourceCalculator.generateActionItems(
          forecast.summary.daily_average,
          { aqi: req.body.current_aqi || 150 },
          null,
          null
        )
      }
    });

  } catch (error) {
    logger.error('Forecast endpoint error:', error);
    res.status(500).json({
      error: 'Forecast failed',
      message: error.message
    });
  }
});

app.post('/api/alerts/send', async (req, res) => {
  try {
    const { recipients, message, priority = 'normal' } = req.body;

    if (!twilioClient) {
      return res.status(503).json({
        error: 'SMS service unavailable',
        message: 'Twilio not configured'
      });
    }

    if (!recipients || !Array.isArray(recipients) || recipients.length === 0) {
      return res.status(400).json({
        error: 'Invalid recipients',
        message: 'Recipients array is required'
      });
    }

    if (!message || message.trim().length === 0) {
      return res.status(400).json({
        error: 'Invalid message',
        message: 'Message content is required'
      });
    }

    const results = [];
    const failedSends = [];

    for (const phoneNumber of recipients) {
      try {
        const smsMessage = await twilioClient.messages.create({
          to: phoneNumber,
          from: process.env.TWILIO_PHONE_NUMBER,
          body: `[MediPulse AI] ${message}`
        });

        results.push({
          to: phoneNumber,
          sid: smsMessage.sid,
          status: 'sent'
        });

        logger.info(`SMS sent successfully to ${phoneNumber}: ${smsMessage.sid}`);

      } catch (smsError) {
        logger.error(`SMS failed to ${phoneNumber}:`, smsError);
        failedSends.push({
          to: phoneNumber,
          error: smsError.message
        });
      }
    }

    res.json({
      success: results.length > 0,
      sent: results.length,
      failed: failedSends.length,
      results,
      failures: failedSends,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Alert sending error:', error);
    res.status(500).json({
      error: 'Alert sending failed',
      message: error.message
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: 'An unexpected error occurred',
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    requested: req.originalUrl,
    available_endpoints: [
      'GET /',
      'GET /api/health',
      'POST /api/predict',
      'POST /api/forecast',
      'POST /api/alerts/send'
    ]
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

const server = app.listen(PORT, () => {
  logger.info(`🚀 MediPulse AI Backend server running on port ${PORT}`);
  logger.info(`📡 FastAPI URL: ${FASTAPI_URL}`);
  logger.info(`📱 Twilio SMS: ${twilioClient ? 'Enabled' : 'Disabled'}`);
  logger.info(`🌐 CORS origin: ${process.env.FRONTEND_URL || 'http://localhost:3000'}`);
});

module.exports = app;