import axios from 'axios';
import { PredictionRequest, PredictionData } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API request failed:', {
      status: error.response?.status,
      message: error.message,
      data: error.response?.data
    });
    
    if (error.response?.status >= 500) {
      throw new Error('Server error. Please try again later.');
    } else if (error.response?.status === 400) {
      throw new Error(error.response.data?.error || 'Invalid request data.');
    } else if (error.response?.status === 404) {
      throw new Error('API endpoint not found.');
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please check your connection.');
    } else {
      throw new Error(error.response?.data?.error || 'An unexpected error occurred.');
    }
  }
);

export const predictPatientLoad = async (request: PredictionRequest): Promise<PredictionData> => {
  const response = await apiClient.post('/api/predict', request);
  return response.data;
};

export const batchPredict = async (requests: PredictionRequest[]): Promise<PredictionData[]> => {
  const response = await apiClient.post('/api/predict/batch', requests);
  return response.data;
};

export const checkHealth = async (): Promise<any> => {
  const response = await apiClient.get('/api/health');
  return response.data;
};

export const getCurrentAQI = async (city: string): Promise<number> => {
  // Mock implementation - in production, integrate with real AQI API
  const mockData: { [key: string]: number } = {
    'Delhi': Math.floor(Math.random() * 200) + 100,
    'Mumbai': Math.floor(Math.random() * 150) + 75,
    'Bangalore': Math.floor(Math.random() * 100) + 50,
    'Chennai': Math.floor(Math.random() * 120) + 60,
    'Kolkata': Math.floor(Math.random() * 180) + 90,
  };
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockData[city] || 100);
    }, 500);
  });
};

export const getCurrentWeather = async (city: string): Promise<{ temperature: number; humidity: number }> => {
  // Mock implementation - in production, integrate with weather API
  const mockData: { [key: string]: { temperature: number; humidity: number } } = {
    'Delhi': { 
      temperature: Math.floor(Math.random() * 20) + 15, 
      humidity: Math.floor(Math.random() * 40) + 40 
    },
    'Mumbai': { 
      temperature: Math.floor(Math.random() * 15) + 20, 
      humidity: Math.floor(Math.random() * 30) + 60 
    },
    'Bangalore': { 
      temperature: Math.floor(Math.random() * 12) + 18, 
      humidity: Math.floor(Math.random() * 25) + 45 
    },
    'Chennai': { 
      temperature: Math.floor(Math.random() * 18) + 22, 
      humidity: Math.floor(Math.random() * 35) + 55 
    },
    'Kolkata': { 
      temperature: Math.floor(Math.random() * 22) + 16, 
      humidity: Math.floor(Math.random() * 40) + 50 
    },
  };
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockData[city] || { temperature: 25, humidity: 50 });
    }, 500);
  });
};

export default apiClient;