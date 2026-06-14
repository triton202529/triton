/**
 * pages/admin/execution-gates.jsx
 * Optional admin dashboard — read-only view of TRITON execution path gate statuses.
 */

import {
  listAllExecutionPathStatuses,
  evaluateExecutionGate,
  explainExecutionDecision,
  EXECUTION_PATHS,
} from '../../lib/executionGate.js';

/**
 * @param {{ environment?: string, paperMode?: boolean }} props
 */
export default function ExecutionGatesPage({ environment = 'development', paperMode = true }) {
  const paths = listAllExecutionPathStatuses();

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: '2rem', maxWidth: 960 }}>
      <h1>TRITON Execution Gates</h1>
      <p>
        THP-002 execution authorization — environment:
        {' '}
        <strong>{environment}</strong>
        {' '}
        | paper:
        {' '}
        <strong>{paperMode ? 'yes' : 'no'}</strong>
      </p>

      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1.5rem' }}>
        <thead>
          <tr>
            <th style={cellHeader}>Path</th>
            <th style={cellHeader}>Default</th>
            <th style={cellHeader}>Risk</th>
            <th style={cellHeader}>Live Decision</th>
            <th style={cellHeader}>Allowed</th>
          </tr>
        </thead>
        <tbody>
          {paths.map((item) => {
            const live = evaluateExecutionGate(item.path, { environment, paperMode });
            return (
              <tr key={item.path}>
                <td style={cell}>{item.path}</td>
                <td style={cell}>{item.defaultDecision}</td>
                <td style={cell}>{item.riskLevel}</td>
                <td style={cell}>{live.decision}</td>
                <td style={cell}>{live.allowed ? 'yes' : 'no'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <section style={{ marginTop: '2rem' }}>
        <h2>Sample: Live Execution</h2>
        <pre style={pre}>
          {explainExecutionDecision(
            evaluateExecutionGate(EXECUTION_PATHS.LIVE_EXECUTION, { environment, liveMode: true }),
          )}
        </pre>
      </section>
    </main>
  );
}

const cellHeader = {
  textAlign: 'left',
  borderBottom: '2px solid #ccc',
  padding: '0.5rem',
};

const cell = {
  borderBottom: '1px solid #eee',
  padding: '0.5rem',
  verticalAlign: 'top',
};

const pre = {
  background: '#f6f8fa',
  padding: '1rem',
  borderRadius: 6,
  overflow: 'auto',
  fontSize: '0.875rem',
};
