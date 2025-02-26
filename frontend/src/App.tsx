import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, Activity, BarChart3, Waves } from 'lucide-react';

interface ChartData {
  Date: string;
  Price: number;
  Moving_Avg: number;
}

interface DecompositionData {
  trend: { date: string; value: number }[];
  seasonality: { date: string; value: number }[];
  residual: { date: string; value: number }[];
}

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString();
  } catch (e) {
    return dateStr;
  }
}

function App() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<{
    movingAverage: ChartData[];
    decomposition: DecompositionData;
  } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/analyze');
        
        // Safely transform moving average data
        const movingAverageData = response.data.moving_average.map((item: any) => ({
          Date: formatDate(item.Date),
          Price: Number(item.Price),
          Moving_Avg: Number(item.Moving_Avg)
        }));

        // Process decomposition data
        const processDecompositionData = (data: Record<string, number>) => {
          return Object.entries(data).map(([date, value]) => ({
            date: formatDate(date),
            value: Number(value)
          }));
        };

        setData({
          movingAverage: movingAverageData,
          decomposition: {
            trend: processDecompositionData(response.data.trend),
            seasonality: processDecompositionData(response.data.seasonality),
            residual: processDecompositionData(response.data.residual)
          }
        });
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  const renderDecompositionChart = (
    data: { date: string; value: number }[],
    color: string
  ) => {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis 
            dataKey="date" 
            stroke="#ffffff" 
            // tick={{ fill: '#ffffff' }}
            tick={false} 
            tickFormatter={(value) => value.split('/')[0]}
          />
          <YAxis stroke="#ffffff" tick={{ fill: '#ffffff' }} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'rgba(19, 19, 26, 0.95)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px',
              color: '#ffffff'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke={color} 
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="min-h-screen p-8">
      <header className="mb-12">
        <h1 className="text-4xl font-bold gradient-text mb-2">Brent Oil Price Analysis</h1>
        <p className="text-gray-400">Interactive visualization of oil price trends and patterns</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass-card p-6">
          <div className="flex items-center mb-6">
            <TrendingUp className="w-6 h-6 text-indigo-500 mr-3" />
            <h2 className="text-xl font-semibold">Price & Moving Average</h2>
          </div>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.movingAverage}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="Date" 
                  stroke="#ffffff" 
                  // tick={{ fill: '#ffffff' }}
                  tick={false} 
                  tickFormatter={(value) => value.split('/')[0]}
                />
                <YAxis stroke="#ffffff" tick={{ fill: '#ffffff' }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(19, 19, 26, 0.95)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    color: '#ffffff'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="Price" 
                  stroke="#6366f1" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="Moving_Avg" 
                  stroke="#8b5cf6" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card p-6">
          <div className="flex items-center mb-6">
            <Activity className="w-6 h-6 text-indigo-500 mr-3" />
            <h2 className="text-xl font-semibold">Trend Analysis</h2>
          </div>
          <div className="h-[400px]">
            {data && renderDecompositionChart(data.decomposition.trend, '#10b981')}
          </div>
        </div>

        <div className="glass-card p-6">
          <div className="flex items-center mb-6">
            <Waves className="w-6 h-6 text-indigo-500 mr-3" />
            <h2 className="text-xl font-semibold">Seasonality Pattern</h2>
          </div>
          <div className="h-[400px]">
            {data && renderDecompositionChart(data.decomposition.seasonality, '#f59e0b')}
          </div>
        </div>

        <div className="glass-card p-6">
          <div className="flex items-center mb-6">
            <BarChart3 className="w-6 h-6 text-indigo-500 mr-3" />
            <h2 className="text-xl font-semibold">Residual Analysis</h2>
          </div>
          <div className="h-[400px]">
            {data && renderDecompositionChart(data.decomposition.residual, '#ef4444')}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;