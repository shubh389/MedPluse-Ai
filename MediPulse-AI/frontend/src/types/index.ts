export interface PredictionRequest {
  city: string;
  date: string;
  aqi: number;
  temperature: number;
  festival?: string;
  outbreak: number;
}

export interface PredictionData {
  predicted_patients: number;
  confidence_interval?: [number, number];
  model_version: string;
  timestamp: string;
  input_summary: {
    city: string;
    aqi: number;
    temperature: number;
    has_festival: boolean;
    outbreak_level: number;
  };
  recommendations?: ResourceRecommendations;
  advisories?: Advisory[];
}

export interface ResourceRecommendations {
  staff: {
    doctors: StaffRecommendation;
    nurses: StaffRecommendation;
    technicians: StaffRecommendation;
  };
  supplies: {
    oxygen: SupplyRecommendation;
    ventilators: SupplyRecommendation;
    beds: SupplyRecommendation;
    ppe: PPERecommendation;
  };
  totalCost: number;
  surgeLevel: 'normal' | 'medium' | 'high' | 'critical';
  actionItems: ActionItem[];
}

export interface StaffRecommendation {
  total: number;
  departments?: { [key: string]: number };
  overtime?: boolean;
  cost: number;
}

export interface SupplyRecommendation {
  required: number;
  recommended: number;
  currentStock: number;
  shortage: number;
  cost: number;
}

export interface PPERecommendation {
  masks: number;
  gloves: number;
  sanitizer: number;
}

export interface Advisory {
  type: 'environmental' | 'capacity' | 'outbreak' | 'general';
  severity: 'info' | 'medium' | 'high' | 'critical';
  message: string;
  actions: string[];
}

export interface ActionItem {
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  action: string;
  timeline: string;
}

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor?: string;
    backgroundColor?: string;
    fill?: boolean;
  }[];
}

export interface City {
  name: string;
  value: string;
}

export interface Festival {
  name: string;
  date: string;
}