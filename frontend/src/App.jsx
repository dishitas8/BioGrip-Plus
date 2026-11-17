
import { useState, useEffect, useMemo } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [form, setForm] = useState({
    crop_type: "Tomato",
    size_cm: 5.7,
    maturity: 0.8,
    temperature_c: 27,
    humidity_percent: 60,
    pressure_kpa: 18,
    deformation_mm: 1.5,
  });

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [error, setError] = useState("");
  const [historyError, setHistoryError] = useState("");

  // Fetch prediction history from FastAPI
  const fetchHistory = async () => {
    try {
      setHistoryError("");

      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error("Could not load prediction history.");
      }

      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Load history when the page opens
  useEffect(() => {
    fetchHistory();
  }, []);

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  // Submit prediction
  const handlePredict = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...form,
          size_cm: Number(form.size_cm),
          maturity: Number(form.maturity),
          temperature_c: Number(form.temperature_c),
          humidity_percent: Number(form.humidity_percent),
          pressure_kpa: Number(form.pressure_kpa),
          deformation_mm: Number(form.deformation_mm),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail?.[0]?.msg ||
          data.error ||
          "Prediction failed"
        );
      }

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);

      // Refresh history after a successful prediction
      await fetchHistory();

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Dashboard calculations
  const totalPredictions = history.length;

  const averageForce = useMemo(() => {
    if (history.length === 0) return "0.000";

    const total = history.reduce(
      (sum, item) => sum + item.predicted_grip_force_n,
      0
    );

    return (total / history.length).toFixed(3);
  }, [history]);

  const latestCrop = history.length > 0
    ? history[0].crop_type
    : "None";

  // Crop-wise prediction counts and average forces
  const cropAnalytics = useMemo(() => {
    const crops = ["Tomato", "Strawberry", "Grapes", "Mushroom"];

    return crops.map((crop) => {
      const records = history.filter(
        (item) => item.crop_type === crop
      );

      const average = records.length
        ? records.reduce(
            (sum, item) => sum + item.predicted_grip_force_n,
            0
          ) / records.length
        : 0;

      return {
        crop,
        count: records.length,
        average: average,
      };
    });
  }, [history]);

  const maxAverage = Math.max(
    ...cropAnalytics.map((item) => item.average),
    0.001
  );

  return (
    <div className="app">

      {/* Navigation */}
      <header className="navbar">
        <h2>BioGrip<span>+</span></h2>
        <p>AI-Powered Soft Robotic Gripper</p>
      </header>

      <main className="container">

        {/* Hero Section */}
        <section className="hero">
          <span className="tag">
            SMART AGRICULTURE • AI • ROBOTICS
          </span>

          <h1>Intelligent Grip Force Prediction</h1>

          <p>
            Predict crop-specific target grip force using simulated
            sensor information and a machine learning model.
          </p>
        </section>

        {/* Prediction Form and Result */}
        <div className="dashboard">

          <section className="panel">
            <h2>Crop & Sensor Inputs</h2>
            <p className="subtext">
              Enter the crop and sensor values.
            </p>

            <form onSubmit={handlePredict}>

              <label>Crop Type</label>

              <select
                name="crop_type"
                value={form.crop_type}
                onChange={handleChange}
              >
                <option>Tomato</option>
                <option>Strawberry</option>
                <option>Grapes</option>
                <option>Mushroom</option>
              </select>

              <div className="form-grid">

                <div>
                  <label>Crop Size (cm)</label>
                  <input
                    name="size_cm"
                    type="number"
                    min="2"
                    max="8"
                    step="0.1"
                    value={form.size_cm}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div>
                  <label>Maturity (0.4–1.0)</label>
                  <input
                    name="maturity"
                    type="number"
                    min="0.4"
                    max="1"
                    step="0.01"
                    value={form.maturity}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div>
                  <label>Temperature (°C)</label>
                  <input
                    name="temperature_c"
                    type="number"
                    min="20"
                    max="35"
                    step="0.1"
                    value={form.temperature_c}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div>
                  <label>Humidity (%)</label>
                  <input
                    name="humidity_percent"
                    type="number"
                    min="40"
                    max="90"
                    step="0.1"
                    value={form.humidity_percent}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div>
                  <label>Pressure (kPa)</label>
                  <input
                    name="pressure_kpa"
                    type="number"
                    min="0"
                    step="0.1"
                    value={form.pressure_kpa}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div>
                  <label>Deformation (mm)</label>
                  <input
                    name="deformation_mm"
                    type="number"
                    min="0"
                    step="0.01"
                    value={form.deformation_mm}
                    onChange={handleChange}
                    required
                  />
                </div>

              </div>

              <button type="submit" disabled={loading}>
                {loading ? "Predicting..." : "Predict Grip Force"}
              </button>

            </form>
          </section>

          {/* Prediction Result */}
          <section className="panel result-panel">

            <h2>AI Prediction</h2>

            <p className="subtext">
              Model output from the BioGrip+ API.
            </p>

            {loading && (
              <p>Generating prediction...</p>
            )}

            {error && (
              <p className="error">{error}</p>
            )}

            {result ? (
              <div className="result">

                <span className="result-label">
                  Predicted Target Grip Force
                </span>

                <h1>
                  {result.predicted_grip_force_n}
                  <small> N</small>
                </h1>

                <p>
                  Crop: <strong>{result.crop}</strong>
                </p>

                <p>
                  Prediction ID: {result.prediction_id}
                </p>

                <p>Model: Random Forest V2</p>

                <p className="notice">
                  {result.note}
                </p>

              </div>
            ) : (
              !loading && !error && (
                <div className="empty">
                  <div className="symbol">◎</div>
                  <h3>Awaiting Prediction</h3>
                  <p>
                    Submit the inputs to see the model output here.
                  </p>
                </div>
              )
            )}

          </section>
        </div>

        {/* Dashboard Summary */}
        <section className="analytics-section">

          <div className="section-heading">
            <span className="tag">PROJECT OVERVIEW</span>
            <h2>BioGrip+ Dashboard</h2>
            <p>Summary of saved model predictions.</p>
          </div>

          <div className="stats-grid">

            <div className="stat-card">
              <p>Total Predictions</p>
              <h2>{historyLoading ? "..." : totalPredictions}</h2>
              <span>Saved records</span>
            </div>

            <div className="stat-card">
              <p>Average Predicted Force</p>
              <h2>{historyLoading ? "..." : averageForce}</h2>
              <span>Newton (N)</span>
            </div>

            <div className="stat-card">
              <p>Most Recent Crop</p>
              <h2>{historyLoading ? "..." : latestCrop}</h2>
              <span>Latest saved prediction</span>
            </div>

          </div>

          {/* Crop Analytics */}
          <div className="analytics-panel">

            <div className="panel-heading">
              <div>
                <h2>Crop-wise Analytics</h2>
                <p className="subtext">
                  Average predicted force by crop type.
                </p>
              </div>
            </div>

            {historyLoading ? (
              <p>Loading analytics...</p>
            ) : (
              <div className="chart-list">

                {cropAnalytics.map((item) => (
                  <div className="chart-row" key={item.crop}>

                    <div className="chart-label">
                      <strong>{item.crop}</strong>
                      <span>
                        {item.count} prediction(s)
                      </span>
                    </div>

                    <div className="bar-background">
                      <div
                        className="bar-fill"
                        style={{
                          width: item.count
                            ? `${(item.average / maxAverage) * 100}%`
                            : "0%",
                        }}
                      />
                    </div>

                    <strong className="chart-value">
                      {item.count
                        ? `${item.average.toFixed(3)} N`
                        : "No data"}
                    </strong>

                  </div>
                ))}

              </div>
            )}

            <p className="chart-note">
              Analytics are calculated from saved simulation-based
              predictions, not physical harvesting measurements.
            </p>

          </div>
        </section>

        {/* Prediction History */}
        <section className="history-section">

          <div className="section-heading">
            <span className="tag">DATABASE RECORDS</span>
            <h2>Prediction History</h2>
            <p>Latest predictions saved in the BioGrip+ database.</p>
          </div>

          <div className="history-panel">

            <div className="history-header">
              <h2>Recent Predictions</h2>

              <button
                type="button"
                className="refresh-button"
                onClick={fetchHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Loading..." : "Refresh History"}
              </button>
            </div>

            {historyError && (
              <p className="error">{historyError}</p>
            )}

            {historyLoading ? (
              <p>Loading prediction records...</p>
            ) : history.length === 0 ? (
              <div className="empty">
                <h3>No Predictions Yet</h3>
                <p>Make your first prediction to see it here.</p>
              </div>
            ) : (

              <div className="table-wrapper">

                <table className="history-table">

                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Crop</th>
                      <th>Size (cm)</th>
                      <th>Maturity</th>
                      <th>Temperature</th>
                      <th>Humidity</th>
                      <th>Pressure</th>
                      <th>Deformation</th>
                      <th>Grip Force (N)</th>
                      <th>Date & Time</th>
                    </tr>
                  </thead>

                  <tbody>

                    {history.map((item) => (
                      <tr key={item.id}>

                        <td>{item.id}</td>
                        <td>{item.crop_type}</td>
                        <td>{item.size_cm}</td>
                        <td>{item.maturity}</td>
                        <td>{item.temperature_c} °C</td>
                        <td>{item.humidity_percent}%</td>
                        <td>{item.pressure_kpa}</td>
                        <td>{item.deformation_mm}</td>

                        <td className="force-cell">
                          {Number(
                            item.predicted_grip_force_n
                          ).toFixed(3)}
                        </td>

                        <td>
                          {new Date(item.created_at).toLocaleString()}
                        </td>

                      </tr>
                    ))}

                  </tbody>

                </table>

              </div>
            )}

          </div>
        </section>

        <footer>
          <p>BioGrip+ | AI-Powered Soft Robotic Gripper</p>
          <p>Simulation-based research prototype</p>
        </footer>

      </main>
    </div>
  );
}

export default App;
