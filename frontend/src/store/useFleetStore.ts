import { create } from 'zustand';
import { Ambulance, Prediction, EmergencyRequest, RouteData, SystemMetrics } from '../types';
import axios from 'axios';

interface FleetState {
  ambulances: Ambulance[];
  predictions: Prediction[];
  emergencies: EmergencyRequest[];
  metrics: SystemMetrics;
  activeRoute: RouteData | null;
  dispatchMode: boolean;
  loading: boolean;
  error: string | null;

  fetchAmbulances: () => Promise<void>;
  fetchPredictions: () => Promise<void>;
  fetchEmergencies: () => Promise<void>;
  fetchMetrics: () => Promise<void>;
  createEmergency: (lat: number, lon: number, priority: string) => Promise<void>;
  resetSystem: () => Promise<void>;
  toggleDispatchMode: () => void;
  fetchRoute: (ambulanceId: number, destinationNode: number) => Promise<void>;
  clearRoute: () => void;
}

const API_BASE = 'http://localhost:8000';

export const useFleetStore = create<FleetState>((set, get) => ({
  ambulances: [],
  predictions: [],
  emergencies: [],
  metrics: { avg_eta: '0.0m', coverage: '0%' },
  activeRoute: null,
  dispatchMode: false,
  loading: false,
  error: null,

  fetchAmbulances: async () => {
    try {
      const res = await axios.get(`${API_BASE}/ambulances`);
      set({ ambulances: res.data });
    } catch (err) {
      set({ error: 'Failed to fetch ambulances' });
    }
  },

  fetchPredictions: async () => {
    try {
      const res = await axios.get(`${API_BASE}/prediction`);
      set({ predictions: res.data });
    } catch (err) {
      set({ error: 'Failed to fetch predictions' });
    }
  },

  fetchEmergencies: async () => {
    try {
      const res = await axios.get(`${API_BASE}/emergencies`);
      set({ emergencies: res.data });
    } catch (err) {
      set({ error: 'Failed to fetch emergencies' });
    }
  },

  fetchMetrics: async () => {
    try {
      const res = await axios.get(`${API_BASE}/metrics`);
      set({ metrics: res.data });
    } catch (err) {
      console.error('Failed to fetch metrics', err);
    }
  },

  createEmergency: async (lat, lon, priority) => {
    set({ loading: true });
    try {
      const res = await axios.post(`${API_BASE}/emergency`, {
        location: [lat, lon],
        priority
      });
      
      const newEmergency: EmergencyRequest = {
        id: res.data.request_id,
        location: [lat, lon],
        priority: priority as any,
        timestamp: new Date().toISOString(),
        assigned_ambulance_id: res.data.assigned_ambulance,
        status: 'dispatched'
      };

      set(state => ({
        emergencies: [newEmergency, ...state.emergencies],
        loading: false
      }));

      // If a route was returned, fetch its full details
      if (res.data.route) {
        await get().fetchRoute(res.data.assigned_ambulance, res.data.route[res.data.route.length - 1]);
      }
    } catch (err) {
      set({ error: 'Failed to create emergency', loading: false });
    }
  },

  resetSystem: async () => {
    set({ loading: true });
    try {
      await axios.post(`${API_BASE}/reset`);
      // Clear local state immediately
      set({ 
        emergencies: [], 
        activeRoute: null, 
        dispatchMode: false,
        loading: false,
        error: null
      });
      // Force refresh data from backend
      await Promise.all([
        get().fetchAmbulances(),
        get().fetchPredictions(),
        get().fetchEmergencies(),
        get().fetchMetrics()
      ]);
    } catch (err) {
      set({ error: 'Failed to reset system', loading: false });
    }
  },

  toggleDispatchMode: () => set(state => ({ dispatchMode: !state.dispatchMode })),

  fetchRoute: async (ambulanceId, destinationNode) => {
    try {
      const res = await axios.get(`${API_BASE}/route`, {
        params: { ambulance_id: ambulanceId, destination_node: destinationNode }
      });
      set({ activeRoute: res.data });
    } catch (err) {
      set({ error: 'Failed to fetch route' });
    }
  },

  clearRoute: () => set({ activeRoute: null }),
}));
