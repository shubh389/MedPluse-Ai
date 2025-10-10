import React from 'react';
import { PredictionData } from '../types';
import ForecastChart from './ForecastChart';
import ResourceCards from './ResourceCards';
import AdvisoryPanel from './AdvisoryPanel';
import SummaryCards from './SummaryCards';

interface DashboardProps {
  prediction: PredictionData | null;
  loading: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({ prediction, loading }) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="loading-spinner mr-3"></div>
        <span className="text-gray-600">Loading dashboard data...</span>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="text-center py-12">
        <div className="max-w-md mx-auto">
          <div className="w-16 h-16 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Prediction Data</h3>
          <p className="text-gray-600 mb-4">Click "New Prediction" to generate a forecast.</p>
        </div>
      </div>
    );
  }

  const mockHistoricalData = {
    labels: ['6 days ago', '5 days ago', '4 days ago', '3 days ago', '2 days ago', 'Yesterday', 'Today', 'Tomorrow'],
    datasets: [
      {
        label: 'Actual Patients',
        data: [120, 135, 142, 158, 149, 167, 0, 0],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
      },
      {
        label: 'Predicted Patients',
        data: [0, 0, 0, 0, 0, 0, prediction.predicted_patients, prediction.predicted_patients + Math.floor(Math.random() * 20 - 10)],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderDash: [5, 5],
        fill: false,
      }
    ]
  };

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <SummaryCards prediction={prediction} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Section */}
        <div className="lg:col-span-2">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold text-gray-900">Patient Load Forecast</h2>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                  <span className="text-gray-600">Actual</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                  <span className="text-gray-600">Predicted</span>
                </div>
              </div>
            </div>
            <ForecastChart data={mockHistoricalData} />
          </div>
        </div>

        {/* Advisory Panel */}
        <div className="lg:col-span-1">
          <AdvisoryPanel 
            advisories={prediction.advisories || []}
            surge={prediction.recommendations?.surgeLevel || 'normal'}
          />
        </div>
      </div>

      {/* Resource Recommendations */}
      {prediction.recommendations && (
        <ResourceCards recommendations={prediction.recommendations} />
      )}

      {/* Data Summary */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <dt className="text-sm font-medium text-gray-500">City</dt>
            <dd className="mt-1 text-sm text-gray-900">{prediction.input_summary.city}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">AQI</dt>
            <dd className="mt-1 text-sm text-gray-900">
              <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                prediction.input_summary.aqi > 300 ? 'bg-red-100 text-red-800' :
                prediction.input_summary.aqi > 200 ? 'bg-yellow-100 text-yellow-800' :
                prediction.input_summary.aqi > 100 ? 'bg-orange-100 text-orange-800' :
                'bg-green-100 text-green-800'
              }`}>
                {prediction.input_summary.aqi}
              </span>
            </dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Temperature</dt>
            <dd className="mt-1 text-sm text-gray-900">{prediction.input_summary.temperature}°C</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Model Version</dt>
            <dd className="mt-1 text-sm text-gray-900">{prediction.model_version}</dd>
          </div>
        </div>
        
        {prediction.confidence_interval && (
          <div className="mt-4 p-3 bg-blue-50 rounded-md">
            <p className="text-sm text-blue-700">
              <span className="font-medium">Confidence Interval:</span> {prediction.confidence_interval[0]} - {prediction.confidence_interval[1]} patients
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;