import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SETUP & PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Battery Prognostics AI", layout="wide")
st.title("🔋 EV Battery Life Predictor (LSTM-PINN)")

# ---------------------------------------------------------
# 2. MODEL DEFINITION
# ---------------------------------------------------------
class LSTMPINN(tf.keras.Model):
    def __init__(self):
        super(LSTMPINN, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True, unroll=True)
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False, unroll=True)
        self.out = tf.keras.layers.Dense(1, activation=None)
        self.alpha = tf.Variable(0.1, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        return self.out(x)

# ---------------------------------------------------------
# 3. LOAD WEIGHTS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = LSTMPINN()
    _ = model(tf.zeros((1, 10, 1))) # Build graph
    try:
        model.load_weights("lstm_pinn_weights.h5")
        return model, True
    except:
        return model, False

model, success = load_model()

if success:
    st.sidebar.success("✅ AI Model Loaded")
else:
    st.sidebar.error("⚠️ Weights not found. Predictions will be simulated.")

# ---------------------------------------------------------
# 4. CREATE TABS FOR DIFFERENT MODES
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📈 Simulator Demo", "✍️ Manual Input (Real-Time)"])

# ==========================================
# TAB 1: ORIGINAL SIMULATOR
# ==========================================
with tab1:
    st.markdown("### Historical Simulation")
    st.info("This view simulates the entire battery life cycle.")
    
    # Generate Dummy Data
    cycles = np.linspace(0, 1, 100)
    c_true = 1.0 - 0.5 * np.sqrt(cycles) + np.random.normal(0, 0.01, 100)

    # Predict
    X_input = []
    for i in range(len(cycles) - 10):
        X_input.append(cycles[i : i + 10])
    X_input = np.array(X_input).reshape(-1, 10, 1)

    if success:
        c_pred = model.predict(X_input).flatten()
    else:
        c_pred = 1.0 - 0.5 * np.sqrt(cycles[10:])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles * 168, c_true, 'gray', alpha=0.5, label='Sensor Data')
    # Pad prediction to match length
    pad = [np.nan] * 10
    ax.plot(cycles * 168, np.concatenate([pad, c_pred]), 'r-', linewidth=2, label='LSTM-PINN')
    ax.legend()
    st.pyplot(fig)

# ==========================================
# TAB 2: MANUAL INPUT (REAL TIME)
# ==========================================
with tab2:
    st.markdown("### ⚡ Live Prediction Mode")
    st.write("Enter the **last 10 capacity readings** (SOH) separated by commas.")
    st.write("Example: `0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.955, 0.95, 0.945`")

    # 1. User Input
    user_input = st.text_area("Paste Data Here:", height=70)

    # 2. Process Input
    if st.button("Predict Next Cycle"):
        try:
            # Convert string "0.9, 0.8..." to float list
            data_list = [float(x.strip()) for x in user_input.split(',')]
            
            if len(data_list) != 10:
                st.error(f"⚠️ We need exactly 10 data points. You provided {len(data_list)}.")
            else:
                # Prepare for LSTM (Batch=1, Window=10, Feat=1)
                input_tensor = np.array(data_list).reshape(1, 10, 1)
                
                # Predict
                if success:
                    prediction = model.predict(input_tensor)[0][0]
                else:
                    prediction = data_list[-1] - 0.01 # Fake fallback
                
                # Show Result
                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Last Measured SOH", f"{data_list[-1]:.4f}")
                c2.metric("🔮 AI Prediction (Next Cycle)", f"{prediction:.4f}", delta=f"{prediction - data_list[-1]:.4f}")

                # Quick Mini-Plot
                fig2, ax2 = plt.subplots(figsize=(6, 2))
                ax2.plot(range(10), data_list, 'bo-', label="History")
                ax2.plot(10, prediction, 'rx', markersize=10, markeredgewidth=3, label="Prediction")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)
                
                # Logic: Is it degrading fast?
                drop = data_list[-1] - prediction
                if drop > 0.02:
                    st.warning("⚠️ Warning: Accelerated degradation detected!")
                else:
                    st.success("✅ Battery health is stable.")

        except ValueError:
            st.error("❌ Invalid Format. Please enter only numbers separated by commas.")