import React from 'react';
import { ResourceRecommendations } from '../types';

interface ResourceCardsProps {
  recommendations: ResourceRecommendations;
}

const ResourceCards: React.FC<ResourceCardsProps> = ({ recommendations }) => {
  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'high':
        return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-green-100 text-green-800 border-green-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* Staff Recommendations */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Staff Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Doctors */}
          <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-blue-900">Doctors</h4>
              <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <p className="text-2xl font-bold text-blue-900">{recommendations.staff.doctors.total}</p>
            <p className="text-sm text-blue-700">Required staff</p>
            {recommendations.staff.doctors.overtime && (
              <div className="mt-2 text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded">
                Overtime needed
              </div>
            )}
            <div className="mt-2 text-sm text-blue-600">
              Cost: ${recommendations.staff.doctors.cost.toLocaleString()}
            </div>
          </div>

          {/* Nurses */}
          <div className="p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-green-900">Nurses</h4>
              <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <p className="text-2xl font-bold text-green-900">{recommendations.staff.nurses.total}</p>
            <p className="text-sm text-green-700">Required staff</p>
            <div className="mt-2 text-sm text-green-600">
              Cost: ${recommendations.staff.nurses.cost.toLocaleString()}
            </div>
          </div>

          {/* Technicians */}
          <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-purple-900">Technicians</h4>
              <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <p className="text-2xl font-bold text-purple-900">{recommendations.staff.technicians.total}</p>
            <p className="text-sm text-purple-700">Required staff</p>
            <div className="mt-2 text-sm text-purple-600">
              Cost: ${recommendations.staff.technicians.cost.toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {/* Supply Recommendations */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Supply Recommendations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Oxygen */}
          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">Oxygen Cylinders</h4>
              <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <p className="text-xl font-bold text-gray-900">{recommendations.supplies.oxygen.recommended}</p>
            <p className="text-sm text-gray-600">Recommended</p>
            <div className="mt-2">
              <p className="text-xs text-gray-500">Current: {recommendations.supplies.oxygen.currentStock}</p>
              {recommendations.supplies.oxygen.shortage > 0 && (
                <p className="text-xs text-red-600">Shortage: {recommendations.supplies.oxygen.shortage}</p>
              )}
            </div>
          </div>

          {/* Ventilators */}
          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">Ventilators</h4>
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <p className="text-xl font-bold text-gray-900">{recommendations.supplies.ventilators.recommended}</p>
            <p className="text-sm text-gray-600">Recommended</p>
            <div className="mt-2">
              <p className="text-xs text-gray-500">Current: {recommendations.supplies.ventilators.currentStock}</p>
              {recommendations.supplies.ventilators.shortage > 0 && (
                <p className="text-xs text-red-600">Shortage: {recommendations.supplies.ventilators.shortage}</p>
              )}
            </div>
          </div>

          {/* Beds */}
          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">Beds</h4>
              <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
              </svg>
            </div>
            <p className="text-xl font-bold text-gray-900">{recommendations.supplies.beds.recommended}</p>
            <p className="text-sm text-gray-600">Recommended</p>
            <div className="mt-2">
              <p className="text-xs text-gray-500">Current: {recommendations.supplies.beds.currentStock}</p>
              {recommendations.supplies.beds.shortage > 0 && (
                <p className="text-xs text-red-600">Shortage: {recommendations.supplies.beds.shortage}</p>
              )}
            </div>
          </div>

          {/* PPE */}
          <div className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">PPE Supplies</h4>
              <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-gray-600">Masks: {recommendations.supplies.ppe.masks}</p>
              <p className="text-sm text-gray-600">Gloves: {recommendations.supplies.ppe.gloves}</p>
              <p className="text-sm text-gray-600">Sanitizer: {recommendations.supplies.ppe.sanitizer}L</p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Items */}
      {recommendations.actionItems && recommendations.actionItems.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Priority Action Items</h3>
          <div className="space-y-3">
            {recommendations.actionItems.slice(0, 5).map((item, index) => (
              <div key={index} className={`p-3 rounded-lg border ${getUrgencyColor(item.priority)}`}>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium">{item.action}</p>
                    <div className="flex items-center space-x-4 mt-1 text-sm opacity-75">
                      <span>Category: {item.category}</span>
                      <span>Timeline: {item.timeline}</span>
                    </div>
                  </div>
                  <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${getUrgencyColor(item.priority)}`}>
                    {item.priority}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResourceCards;