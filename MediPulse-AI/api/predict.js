const axios = require('axios');

// Cache for predictions to reduce ML service calls
let predictionCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Resource calculator class
class ResourceCalculator {
  calculateStaffing(patientCount, surgeProbability = 0) {
    const baseNurseRatio = 4; // 1 nurse per 4 patients
    const baseDoctorRatio = 8; // 1 doctor per 8 patients
    
    const surgeMultiplier = 1 + (surgeProbability * 0.5);
    
    return {
      nurses: Math.ceil((patientCount / baseNurseRatio) * surgeMultiplier),
      doctors: Math.ceil((patientCount / baseDoctorRatio) * surgeMultiplier),
      beds: Math.ceil(patientCount * 1.2), // 20% buffer
    };
  }

  calculateSupplies(patientCount, weatherConditions) {
    const baseSupplies = {
      masks: patientCount * 2,
      gloves: patientCount * 4,
      sanitizer: Math.ceil(patientCount / 10),
      medicines: patientCount * 1.5,
    };

    // Adjust for weather conditions
    if (weatherConditions === 'severe' || weatherConditions === 'extreme') {
      Object.keys(baseSupplies).forEach(key => {
        baseSupplies[key] = Math.ceil(baseSupplies[key] * 1.3);
      });
    }

    return baseSupplies;
  }

  generateAdvisories(prediction, inputData) {
    const advisories = [];
    
    if (prediction > 150) {
      advisories.push({
        type: 'alert',
        message: 'High patient load expected. Consider activating surge protocols.',
        priority: 'high'
      });
    }

    if (inputData.air_quality_index > 200) {
      advisories.push({
        type: 'health',
        message: 'Poor air quality may increase respiratory cases.',
        priority: 'medium'
      });
    }

    if (inputData.local_events !== 'none') {
      advisories.push({
        type: 'event',
        message: `Local event (${inputData.local_events}) may affect patient patterns.`,
        priority: 'medium'
      });
    }

    return advisories;
  }
}

const resourceCalculator = new ResourceCalculator();

module.exports = async (req, res) => {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const inputData = req.body;
    
    // Validate required fields
    const requiredFields = ['date', 'day_of_week', 'month', 'season', 'weather_conditions', 'city'];
    for (const field of requiredFields) {
      if (!inputData[field]) {
        return res.status(400).json({ error: `Missing required field: ${field}` });
      }
    }

    // Check cache first
    const cacheKey = JSON.stringify(inputData);
    const cached = predictionCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
      return res.json(cached.data);
    }

    // Call ML service
    const mlServiceUrl = process.env.ML_SERVICE_URL || '/api/ml';
    let mlResponse;
    
    try {
      mlResponse = await axios.post(`${mlServiceUrl}/predict`, inputData, {
        timeout: 25000,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    } catch (mlError) {
      console.error('ML Service Error:', mlError.message);
      
      // Fallback prediction if ML service fails
      const fallbackPrediction = 100 + Math.floor(Math.random() * 50);
      const staffing = resourceCalculator.calculateStaffing(fallbackPrediction);
      const supplies = resourceCalculator.calculateSupplies(fallbackPrediction, inputData.weather_conditions);
      
      return res.json({
        predicted_patients: fallbackPrediction,
        confidence_score: 0.6,
        recommendations: {
          ...staffing,
          supplies,
          surgeLevel: fallbackPrediction > 120 ? 'high' : 'normal'
        },
        advisories: [{
          type: 'warning',
          message: 'Using fallback prediction due to ML service unavailability',
          priority: 'medium'
        }],
        fallback: true,
        timestamp: new Date().toISOString()
      });
    }

    const prediction = mlResponse.data.predicted_patients;
    const confidenceScore = mlResponse.data.confidence_score || 0.85;

    // Calculate recommendations
    const staffing = resourceCalculator.calculateStaffing(prediction, confidenceScore < 0.7 ? 0.3 : 0);
    const supplies = resourceCalculator.calculateSupplies(prediction, inputData.weather_conditions);
    const advisories = resourceCalculator.generateAdvisories(prediction, inputData);

    const result = {
      ...mlResponse.data,
      recommendations: {
        ...staffing,
        supplies,
        surgeLevel: prediction > 150 ? 'high' : prediction > 100 ? 'medium' : 'normal'
      },
      advisories,
      input_summary: {
        city: inputData.city,
        date: inputData.date,
        aqi: inputData.air_quality_index || 50,
        temperature: inputData.temperature || 25,
        weather: inputData.weather_conditions
      },
      timestamp: new Date().toISOString()
    };

    // Cache the result
    predictionCache.set(cacheKey, {
      data: result,
      timestamp: Date.now()
    });

    // Clean old cache entries
    if (predictionCache.size > 100) {
      const entries = Array.from(predictionCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      entries.slice(0, 50).forEach(([key]) => predictionCache.delete(key));
    }

    res.json(result);

  } catch (error) {
    console.error('Prediction API Error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
};