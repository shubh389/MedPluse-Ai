import React from 'react';
import { Advisory } from '../types';

interface AdvisoryPanelProps {
  advisories: Advisory[];
  surge: string;
}

const AdvisoryPanel: React.FC<AdvisoryPanelProps> = ({ advisories, surge }) => {
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return (
          <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.268 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        );
      case 'high':
        return (
          <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'medium':
        return (
          <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'high':
        return 'bg-orange-50 border-orange-200 text-orange-800';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800';
    }
  };

  const getSurgeColor = (surgeLevel: string) => {
    switch (surgeLevel) {
      case 'critical':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'high':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      default:
        return 'bg-green-100 text-green-800 border-green-300';
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border h-fit">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Health Advisories</h3>
        <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getSurgeColor(surge)}`}>
          {surge.charAt(0).toUpperCase() + surge.slice(1)} Alert
        </div>
      </div>

      <div className="space-y-4">
        {advisories.length === 0 ? (
          <div className="text-center py-8">
            <div className="w-12 h-12 mx-auto mb-3 bg-gray-100 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-gray-500 text-sm">No active advisories</p>
            <p className="text-gray-400 text-xs mt-1">All systems normal</p>
          </div>
        ) : (
          advisories.map((advisory, index) => (
            <div key={index} className={`p-4 rounded-lg border ${getSeverityColor(advisory.severity)}`}>
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 mt-1">
                  {getSeverityIcon(advisory.severity)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="text-xs font-medium uppercase tracking-wide opacity-75">
                      {advisory.type}
                    </span>
                    <span className="text-xs px-2 py-1 bg-white bg-opacity-50 rounded">
                      {advisory.severity}
                    </span>
                  </div>
                  <p className="text-sm font-medium mb-2">{advisory.message}</p>
                  {advisory.actions && advisory.actions.length > 0 && (
                    <div className="mt-2">
                      <p className="text-xs font-medium mb-1 opacity-75">Recommended Actions:</p>
                      <ul className="text-xs space-y-1">
                        {advisory.actions.map((action, actionIndex) => (
                          <li key={actionIndex} className="flex items-start">
                            <span className="inline-block w-1 h-1 bg-current rounded-full mt-2 mr-2 flex-shrink-0"></span>
                            <span>{action}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Default Health Tips */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-900 mb-2">General Health Tips</h4>
        <div className="text-xs text-gray-600 space-y-1">
          <p>• Monitor air quality levels regularly</p>
          <p>• Maintain adequate PPE inventory</p>
          <p>• Update emergency contact lists monthly</p>
          <p>• Review surge protocols quarterly</p>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-xs text-yellow-800">
          <span className="font-medium">Disclaimer:</span> These are AI-generated advisory suggestions for decision support. 
          Always follow established hospital protocols and clinical judgment.
        </p>
      </div>
    </div>
  );
};

export default AdvisoryPanel;