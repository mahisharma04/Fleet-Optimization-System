export interface Ambulance {
  id: number;
  lat: number;
  lon: number;
  status: 'idle' | 'busy' | 'transit';
  destination?: number;
}

export interface Prediction {
  cluster_id: number;
  lat: number;
  lon: number;
  score: number;
}

export interface EmergencyRequest {
  id: number;
  location: [number, number];
  priority: 'high' | 'medium' | 'low';
  timestamp: string;
  assigned_ambulance_id?: number;
  status: 'pending' | 'dispatched' | 'completed';
}

export interface RouteData {
  route: number[];
  path_coords: [number, number][];
  eta: number;
}

export interface SystemMetrics {
  avg_eta: string;
  coverage: string;
}
