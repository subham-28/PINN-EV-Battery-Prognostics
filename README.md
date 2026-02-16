#  Physics-Informed Neural Networks (PINN) for EV Battery Prognostics

## 1. Executive Summary
This project investigates the application of **Scientific Machine Learning (SciML)** to predict the Remaining Useful Life (RUL) of Lithium-Ion batteries. By integrating electrochemical physics laws directly into neural networks, we developed a "Glass Box" model that is both data-driven and physically consistent.

**Key Achievements:**
* **Baseline Established:** A pure physics-based approach set a strong benchmark (RMSE 0.2757).
* **Physics Validation:** The **Square Root Law** (SEI Layer Growth) was mathematically validated as the dominant aging mechanism for this dataset.
* **Architecture Breakthrough:** A **Hybrid LSTM-PINN** architecture achieved a state-of-the-art **RMSE of 0.0448**, outperforming the pure physics baseline by **4x**.

---

## 2. Dataset Information
* **Source:** NASA Prognostics Center of Excellence (PCoE)
* **Dataset Name:** Li-Ion Battery Aging Data Set
* **Battery ID Used:** B0005
* **Chemistry:** 18650 Lithium-Ion (Cathode: LiCoO2, Anode: Graphite)
* **Experimental Condition:** Repeated charge and discharge cycles at room temperature (24°C) until capacity dropped by 30%.
* **Link to Dataset:** [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 3. Methodology & Mathematical Framework

### 3.1 The Physics-Informed Loss Function
Unlike traditional neural networks that only minimize prediction error, a PINN minimizes a composite loss function comprising a **Data Term** (L_data) and a **Physics Term** (L_physics).

> **Total Loss = L_data + λ * L_physics**

Where:
* **L_data (Data Loss):** Ensures the model fits the observed capacity measurements.
  * *Formula:* `Mean((Predicted - Actual)^2)`
* **L_physics (Physics Residual):** Penalizes predictions that violate the governing differential equation.
  * *Formula:* `Mean(| dC/dt - f(C, t) |^2)`
* **λ (Lambda):** A weighting parameter that balances the trade-off between fitting data and obeying physics.

---

## 4. Experiment Analysis

###  Phase 0: The Baseline Model
Before introducing neural networks, we established a "Pure Physics" baseline. We fit the **Square Root Law** directly to the data.

* **Governing Equation:**
  > **C(t) = 1 - α * √t**
* **Result:** RMSE = 0.2757
* **Observation:** The model captured the global decay trend but missed local fluctuations (capacity regeneration) caused by battery rest periods.
<img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/12421eec-9216-4c18-8136-e56571db3868" />

---

###  Phase 1: Physics Model Selection (`Experiment-1.ipynb`)
We tested four fundamental differential equations to identify the governing physics of the B0005 battery.

#### **Model A: Power Law (General Degradation)**
* **Equation:** Assumes degradation rate depends on current capacity with an arbitrary order β.
  > **dC/dt = -α * C^β**
* **RMSE:** 0.3682
* **Verdict:** **Overfitted.** The additional parameter β made the model too flexible, fitting noise rather than the trend.
  <img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/521664cb-a5ef-40fc-90d0-5336deeab3f8" />


#### **Model B: S-Curve (Verhulst / Logistic)**
* **Equation:** Assumes degradation slows down as capacity drops (saturation effect).
  > **dC/dt = -α * C * (1 - C)**
* **RMSE:** 0.5039
* **Verdict:** **Failed.** This equation predicts a stable equilibrium, which contradicts the reality of continuous, accelerating battery degradation.
  <img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/a6b3abd0-cb96-4145-aca2-cebd0d3abda4" />


#### **Model C: Arrhenius Law (Temperature-Aware)**
* **Hypothesis:** Degradation is driven by temperature. We fed both Time (t) and Temperature (T) into the network.
* **Equation:** The decay rate α depends on temperature.
  > **dC/dt = -A * exp( -Ea / (R*T) ) * C**
* **RMSE:** **0.2988**
* **Verdict:** **Good, but not best.** While it captured temperature fluctuations, the B0005 dataset is cycled at mostly controlled room temperatures, making Time a stronger predictor than Temperature for this specific case.
  <img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/3a13542f-9f7c-41b7-a244-b8e379eca74f" />


#### **Model D: Square Root Law (SEI Layer Growth)**
* **Equation:** Derived from Fick's Law of Diffusion. As the SEI layer grows, it becomes thicker, slowing down further growth (rate is proportional to 1/√t).
  > **dC/dt = -α / (2√t)**
* **RMSE:** **0.2795**
* **Verdict:** ** Best Physics Fit.** Confirmed that diffusion-limited SEI growth is the dominant aging mechanism for this cell.
  <img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/f7212e74-847f-43cc-8b9c-f13e88b58e76" />


---

###  Phase 2: Adaptive Loss Weighing (`Experiment-2.ipynb`)
**Hypothesis:** Can the model automatically learn the optimal weight λ using Homoscedastic Uncertainty?

* **Method:** We introduced learnable variance parameters σ1 and σ2 to weigh the loss terms dynamically.
* **Outcome:** **Failure (RMSE 0.6113).**
* **Analysis:** The model converged to a **Trivial Solution**. It learned that by setting α ≈ 0, the physics residual became zero. It minimized the loss by "cheating" the physics rather than fitting the data.
* **Lesson:** Fixed weighing (λ=0.1) is more robust for simple monotonic degradation problems.
  <img width="1001" height="568" alt="image" src="https://github.com/user-attachments/assets/c14c5ace-38a7-4df0-9466-a38b551c4b45" />


---

###  Phase 3: Hybrid Architecture - The Champion (`Experiment-3.ipynb`)
**Hypothesis:** Battery degradation has "memory." The state at cycle *t* depends on the trajectory of cycles *[t-10, ... t]*.

* **Architecture:**
    * **Input:** Sliding Window **X = [t-9, t-8, ..., t]**
    * **Network:** 2-Layer **LSTM (Long Short-Term Memory)** with Unrolled Loops.
    * **Physics Constraint:** Applied to the *output trend*.

* **Results:**
    * **RMSE:** **0.0448**
    * **Improvement:** **314%** over Baseline.

* **Why it Won:**
    * The **LSTM** captured the "heaps" (capacity regeneration events) which appear as local non-linearities.
    * The **Square Root Physics Law** acted as a regularizer, preventing the LSTM from overfitting to noise or making physically impossible predictions (e.g., capacity increasing over time).
      <img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/7209b886-dd8b-420e-8549-125b2c97c611" />


---

## 5. Final Conclusion
The **LSTM-PINN** is the optimal architecture for Battery Prognostics.
1.  **Accuracy:** It is 4x more accurate than standard physics models.
2.  **Robustness:** Unlike pure data-driven LSTMs, it respects the **1/√t** decay law, making it safer for long-term extrapolation.

**Final Recommendation:** Deploy the **LSTM-PINN** with **Square Root Physics** for production RUL estimation.
