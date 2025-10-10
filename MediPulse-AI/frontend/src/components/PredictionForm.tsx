import React, { useState, useEffect } from 'react';
import { PredictionRequest, City, Festival } from '../types';
import { getCurrentAQI, getCurrentWeather } from '../services/api';

interface PredictionFormProps {
  onSubmit: (request: PredictionRequest) => Promise<void>;
  loading: boolean;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState<PredictionRequest>({
    city: 'Delhi',
    date: new Date().toISOString().split('T')[0],
    aqi: 150,
    temperature: 25,
    outbreak: 0
  });

  const [loadingExternalData, setLoadingExternalData] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const cities: City[] = [
    { name: 'Delhi', value: 'Delhi' },
    { name: 'Mumbai', value: 'Mumbai' },
    { name: 'Bangalore', value: 'Bangalore' },
    { name: 'Chennai', value: 'Chennai' },
    { name: 'Kolkata', value: 'Kolkata' },
  ];

  const festivals: Festival[] = [
    { name: 'Diwali', date: '2025-11-02' },
    { name: 'Holi', date: '2025-03-14' },
    { name: 'Eid', date: '2025-04-10' },
    { name: 'Christmas', date: '2025-12-25' },
    { name: 'Dussehra', date: '2025-10-12' },
  ];

  const outbreakLevels = [
    { value: 0, label: 'No Outbreak', description: 'Normal conditions' },
    { value: 1, label: 'Localized', description: 'Small area affected' },
    { value: 2, label: 'Regional', description: 'City-wide impact' },
    { value: 3, label: 'Epidemic', description: 'Multi-city spread' },
  ];

  // Auto-fetch current environmental data when city changes
  useEffect(() => {
    const fetchCurrentData = async () => {
      if (!formData.city) return;
      
      setLoadingExternalData(true);
      try {
        const [aqi, weather] = await Promise.all([
          getCurrentAQI(formData.city),
          getCurrentWeather(formData.city)
        ]);

        setFormData(prev => ({
          ...prev,
          aqi: Math.round(aqi),
          temperature: Math.round(weather.temperature * 10) / 10
        }));
      } catch (error) {
        console.error('Failed to fetch current data:', error);
      } finally {
        setLoadingExternalData(false);
      }
    };

    fetchCurrentData();
  }, [formData.city]);

  const validateForm = (): boolean => {
    const newErrors: {[key: string]: string} = {};

    if (!formData.city.trim()) {
      newErrors.city = 'City is required';
    }

    if (!formData.date) {
      newErrors.date = 'Date is required';
    } else {
      const selectedDate = new Date(formData.date);
      const today = new Date();
      const maxDate = new Date();
      maxDate.setDate(today.getDate() + 30); // Allow up to 30 days in the future

      if (selectedDate > maxDate) {
        newErrors.date = 'Date cannot be more than 30 days in the future';
      }
    }

    if (formData.aqi < 0 || formData.aqi > 500) {
      newErrors.aqi = 'AQI must be between 0 and 500';
    }

    if (formData.temperature < -50 || formData.temperature > 60) {
      newErrors.temperature = 'Temperature must be between -50°C and 60°C';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await onSubmit(formData);
    } catch (error) {
      console.error('Form submission error:', error);
    }
  };

  const handleInputChange = (field: keyof PredictionRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = {...prev};
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  const getAQIColor = (aqi: number) => {
    if (aqi > 300) return 'text-red-600';
    if (aqi > 200) return 'text-orange-600';
    if (aqi > 100) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getAQILabel = (aqi: number) => {
    if (aqi > 300) return 'Hazardous';
    if (aqi > 200) return 'Very Unhealthy';
    if (aqi > 150) return 'Unhealthy';
    if (aqi > 100) return 'Unhealthy for Sensitive Groups';
    if (aqi > 50) return 'Moderate';
    return 'Good';
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white p-8 rounded-lg shadow-sm border">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900">New Patient Load Prediction</h2>
          <p className="text-gray-600 mt-2">
            Enter the parameters below to generate a forecast for hospital patient admissions.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* City Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              City *
            </label>
            <select
              value={formData.city}
              onChange={(e) => handleInputChange('city', e.target.value)}
              className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.city ? 'border-red-300' : 'border-gray-300'
              }`}
            >
              {cities.map(city => (
                <option key={city.value} value={city.value}>
                  {city.name}
                </option>
              ))}
            </select>
            {errors.city && <p className="text-red-600 text-sm mt-1">{errors.city}</p>}
          </div>

          {/* Date Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Date *
            </label>
            <input
              type="date"
              value={formData.date}
              onChange={(e) => handleInputChange('date', e.target.value)}
              min={new Date().toISOString().split('T')[0]}
              max={new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]}
              className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                errors.date ? 'border-red-300' : 'border-gray-300'
              }`}
            />
            {errors.date && <p className="text-red-600 text-sm mt-1">{errors.date}</p>}
          </div>

          {/* Environmental Data */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* AQI */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Air Quality Index (AQI) *
                {loadingExternalData && (
                  <span className="ml-2 text-xs text-blue-600">(Auto-updating...)</span>
                )}
              </label>
              <input
                type="number"
                value={formData.aqi}
                onChange={(e) => handleInputChange('aqi', parseInt(e.target.value) || 0)}
                min="0"
                max="500"
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.aqi ? 'border-red-300' : 'border-gray-300'
                }`}
              />
              <div className="mt-1 flex justify-between items-center">
                <span className={`text-sm font-medium ${getAQIColor(formData.aqi)}`}>
                  {getAQILabel(formData.aqi)}
                </span>
                <span className="text-xs text-gray-500">Range: 0-500</span>
              </div>
              {errors.aqi && <p className="text-red-600 text-sm mt-1">{errors.aqi}</p>}
            </div>

            {/* Temperature */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Temperature (°C) *
                {loadingExternalData && (
                  <span className="ml-2 text-xs text-blue-600">(Auto-updating...)</span>
                )}
              </label>
              <input
                type="number"
                value={formData.temperature}
                onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value) || 0)}
                step="0.1"
                min="-50"
                max="60"
                className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                  errors.temperature ? 'border-red-300' : 'border-gray-300'
                }`}
              />
              <div className="mt-1 text-xs text-gray-500">Range: -50°C to 60°C</div>
              {errors.temperature && <p className="text-red-600 text-sm mt-1">{errors.temperature}</p>}
            </div>
          </div>

          {/* Festival Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Festival (Optional)
            </label>
            <select
              value={formData.festival || ''}
              onChange={(e) => handleInputChange('festival', e.target.value || undefined)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">No Festival</option>
              {festivals.map(festival => (
                <option key={festival.name} value={festival.name}>
                  {festival.name} ({festival.date})
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Select if there's a major festival on or around the prediction date
            </p>
          </div>

          {/* Outbreak Level */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Outbreak Level *
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {outbreakLevels.map(level => (
                <label key={level.value} className="flex items-center p-3 border rounded-md cursor-pointer hover:bg-gray-50">
                  <input
                    type="radio"
                    name="outbreak"
                    value={level.value}
                    checked={formData.outbreak === level.value}
                    onChange={(e) => handleInputChange('outbreak', parseInt(e.target.value))}
                    className="mr-3"
                  />
                  <div>
                    <div className="font-medium text-sm">{level.label}</div>
                    <div className="text-xs text-gray-500">{level.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={() => {
                setFormData({
                  city: 'Delhi',
                  date: new Date().toISOString().split('T')[0],
                  aqi: 150,
                  temperature: 25,
                  outbreak: 0
                });
                setErrors({});
              }}
              className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Reset
            </button>
            <button
              type="submit"
              disabled={loading || loadingExternalData}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {loading && <div className="loading-spinner"></div>}
              <span>{loading ? 'Generating Prediction...' : 'Generate Prediction'}</span>
            </button>
          </div>
        </form>

        {/* Help Text */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Tips for Accurate Predictions</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• AQI and temperature are auto-filled with current data for the selected city</li>
            <li>• Select festivals that fall within ±3 days of your prediction date</li>
            <li>• Outbreak levels reflect current public health status in the region</li>
            <li>• Predictions are most accurate for dates within the next 7 days</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;