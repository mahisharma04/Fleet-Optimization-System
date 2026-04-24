import { useFleetStore } from '../../store/useFleetStore';
import { Activity, ShieldAlert, Truck, Timer, MapPin, RefreshCw, MousePointer2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const { ambulances, emergencies, createEmergency, resetSystem, dispatchMode, toggleDispatchMode, loading, metrics } = useFleetStore();

  const handleRandomEmergency = () => {
    const lat = 18.51957 + (Math.random() - 0.5) * 0.05;
    const lon = 73.85529 + (Math.random() - 0.5) * 0.05;
    createEmergency(lat, lon, 'high');
  };

  const stats = [
    { label: 'Active Calls', value: emergencies.length, icon: ShieldAlert, color: 'text-red-500' },
    { label: 'Fleet Size', value: ambulances.length, icon: Truck, color: 'text-blue-500' },
    { label: 'Avg ETA', value: metrics.avg_eta, icon: Timer, color: 'text-yellow-500' },
    { label: 'Coverage', value: metrics.coverage, icon: Activity, color: 'text-green-500' },
  ];

  const dummyChartData = [
    { name: '10:00', value: 4 },
    { name: '11:00', value: 7 },
    { name: '12:00', value: 5 },
    { name: '13:00', value: 9 },
    { name: '14:00', value: 6 },
  ];

  return (
    <div className="flex flex-col h-full bg-background border-l border-border w-96 overflow-y-auto">
      <div className="p-6 space-y-6">
        <header className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight">Fleet Control</h1>
          <p className="text-sm text-muted-foreground">Pune Smart City Operations</p>
        </header>

        <button
          onClick={handleRandomEmergency}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-destructive text-destructive-foreground hover:bg-destructive/90 h-11 px-8 rounded-md font-medium transition-colors disabled:opacity-50"
        >
          <ShieldAlert className="w-5 h-5" />
          {loading ? 'Dispatching...' : 'SIGNAL RANDOM EMERGENCY'}
        </button>

        <button
          onClick={toggleDispatchMode}
          className={`w-full flex items-center justify-center gap-2 h-11 px-8 rounded-md font-medium transition-all border-2 ${
            dispatchMode 
              ? 'bg-blue-600 border-blue-400 text-white animate-pulse' 
              : 'bg-background border-border text-muted-foreground hover:border-blue-500'
          }`}
        >
          <MousePointer2 className="w-5 h-5" />
          {dispatchMode ? 'CLICK MAP TO DISPATCH' : 'ENABLE CLICK-TO-DISPATCH'}
        </button>

        <button
          onClick={() => resetSystem()}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-secondary text-secondary-foreground hover:bg-secondary/80 h-10 px-8 rounded-md text-sm font-medium transition-colors border border-border disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          RESET ALL SYSTEMS
        </button>

        <div className="grid grid-cols-2 gap-4">
          {stats.map((stat) => (
            <div key={stat.label} className="p-4 rounded-xl border border-border bg-card">
              <stat.icon className={`w-5 h-5 mb-2 ${stat.color}`} />
              <p className="text-2xl font-bold">{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        <section className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Recent Emergencies</h2>
          <div className="space-y-3">
            {emergencies.length === 0 ? (
              <p className="text-sm text-muted-foreground italic">No active requests</p>
            ) : (
              emergencies.slice(0, 5).map((e) => (
                <div key={e.id} className="p-3 rounded-lg border border-border bg-card text-sm space-y-1">
                  <div className="flex justify-between items-start">
                    <span className="font-semibold">Request #{e.id}</span>
                    <span className="px-2 py-0.5 rounded-full bg-red-500/10 text-red-500 text-[10px] font-bold uppercase">
                      {e.priority}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-muted-foreground">
                    <MapPin className="w-3 h-3" />
                    <span>{e.location[0].toFixed(4)}, {e.location[1].toFixed(4)}</span>
                  </div>
                  <p className="text-xs">
                    Status:{' '}
                    <span className={`capitalize ${
                      e.status === 'dispatched' ? 'text-yellow-500' : 
                      e.status === 'pending' ? 'text-blue-500' : 
                      e.status === 'completed' ? 'text-green-500' : 'text-gray-500'
                    }`}>
                      {e.status}
                    </span>
                  </p>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Demand Trend</h2>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dummyChartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" />
                <XAxis dataKey="name" hide />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ background: '#1a1a1a', border: '1px solid #333', fontSize: '12px' }}
                />
                <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Dashboard;
