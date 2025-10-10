import React from 'react';
import { PredictionData } from '../types';

interface SummaryCardsProps {
  prediction: PredictionData;
}

const SummaryCards: React.FC<SummaryCardsProps> = ({ prediction }) => {
  const getSurgeLevel = () => {
    const patients = prediction.predicted_patients;
    if (patients > 250) return { level: 'Critical', color: 'bg-red-100 text-red-800 border-red-200' };
    if (patients > 180) return { level: 'High', color: 'bg-orange-100 text-orange-800 border-orange-200' };
    if (patients > 120) return { level: 'Medium', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' };
    return { level: 'Normal', color: 'bg-green-100 text-green-800 border-green-200' };
  };

  const surge = getSurgeLevel();

  const cards = [
    {
      title: 'Predicted Patients',
      value: prediction.predicted_patients,
      unit: 'patients',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      ),
      color: 'bg-blue-50 text-blue-700 border-blue-200'
    },
    {
      title: 'Surge Level',
      value: surge.level,
      unit: '',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
        </svg>
      ),
      color: surge.color
    },
    {
      title: 'AQI Level',
      value: prediction.input_summary.aqi,
      unit: 'AQI',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        </svg>
      ),
      color: prediction.input_summary.aqi > 200 ? 'bg-red-50 text-red-700 border-red-200' : 'bg-gray-50 text-gray-700 border-gray-200'
    },
    {
      title: 'Total Cost',
      value: prediction.recommendations?.totalCost ? `$${Math.round(prediction.recommendations.totalCost).toLocaleString()}` : 'N/A',
      unit: '',
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
        </svg>
      ),
      color: 'bg-green-50 text-green-700 border-green-200'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => (
        <div key={index} className={`p-6 rounded-lg border ${card.color}`}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium opacity-75">{card.title}</p>
              <p className="text-2xl font-bold">
                {card.value} {card.unit && <span className="text-sm font-normal">{card.unit}</span>}
              </p>
            </div>
            <div className="opacity-75">
              {card.icon}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default SummaryCards;