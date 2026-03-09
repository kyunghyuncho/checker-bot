import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Game } from './pages/Game';
import { DataInspector } from './pages/DataInspector';
import { BrainCircuit, Database } from 'lucide-react';
import './index.css';

// Educational Layout Component
const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  
  return (
    <div className="app-container">
      <header className="panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.75rem' }}>Checkers Deep Learning</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Interactive educational platform for training a two-headed CNN
          </p>
        </div>
        
        <nav style={{ display: 'flex', gap: '1rem' }}>
          <Link 
            to="/" 
            style={{ 
              display: 'flex', alignItems: 'center', gap: '0.5rem', 
              textDecoration: 'none', 
              color: location.pathname === '/' ? 'var(--accent-blue)' : 'var(--text-primary)',
              fontWeight: location.pathname === '/' ? '600' : '400'
            }}
          >
            <BrainCircuit size={20} /> Play & Train
          </Link>
          <Link 
            to="/inspector" 
            style={{ 
              display: 'flex', alignItems: 'center', gap: '0.5rem', 
              textDecoration: 'none',
              color: location.pathname === '/inspector' ? 'var(--accent-purple)' : 'var(--text-primary)',
              fontWeight: location.pathname === '/inspector' ? '600' : '400'
            }}
          >
            <Database size={20} /> Data Inspector
          </Link>
        </nav>
      </header>

      <main>
        {children}
      </main>
    </div>
  );
};

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Game />} />
          <Route path="/inspector" element={<DataInspector />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
