import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import { PredictionData, PredictionRequest } from './types';
import { predictPatientLoad } from './services/api';

const App: React.FC = () => {
  const [currentPrediction, setCurrentPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'predict'>('dashboard');

  // Load initial prediction on app start
  useEffect(() => {
    const loadInitialPrediction = async () => {
      try {
        setLoading(true);
        const defaultRequest: PredictionRequest = {
          city: 'Delhi',
          date: new Date().toISOString().split('T')[0],
          aqi: 150,
          temperature: 25,
          outbreak: 0
        };
        const prediction = await predictPatientLoad(defaultRequest);
        setCurrentPrediction(prediction);
      } catch (error) {
        console.error('Failed to load initial prediction:', error);
      } finally {
        setLoading(false);
      }
    };

    loadInitialPrediction();
  }, []);

  const handleNewPrediction = async (request: PredictionRequest) => {
    try {
      setLoading(true);
      const prediction = await predictPatientLoad(request);
      setCurrentPrediction(prediction);
      setActiveTab('dashboard'); // Switch to dashboard to show results
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">MP</span>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">MediPulse AI</h1>
                <p className="text-sm text-gray-500">Hospital Load Forecasting Dashboard</p>
              </div>
            </div>
            
            {/* Navigation */}
            <nav className="flex space-x-4">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'dashboard'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('predict')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'predict'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                New Prediction
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-lg flex items-center space-x-3">
              <div className="loading-spinner"></div>
              <span>Processing prediction...</span>
            </div>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <Dashboard prediction={currentPrediction} loading={loading} />
        )}

        {activeTab === 'predict' && (
          <PredictionForm 
            onSubmit={handleNewPrediction}
            loading={loading}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">
              © 2025 MediPulse AI. All rights reserved.
            </p>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <span>Last updated: {new Date().toLocaleString()}</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>System Online</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;