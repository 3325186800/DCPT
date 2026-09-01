## 📊 Dataset Configurations

This dataset contains 6 independent data subsets designed to comprehensively cover diverse physical scenarios, ranging from ideal direct links to complex indoor and outdoor environments. Each subset consists of 5,000 radio frequency samples (10 devices, 500 samples per device).

The table below details the specific acquisition configurations for each subset constructed in this RFF dataset, including the scenario type, channel conditions, device motion state, and collection date:

| Subset | Scenario | Channel Conditions | Motion State | Date |
| :---: | :---: | :---: | :---: | :---: |
| **D1** | Direct Link | - | Stationary | 2025-10-02 |
| **I1** | Indoor | LOS | Stationary | 2025-10-04 |
| **I2** | Indoor | LOS | Stationary | 2025-11-12 |
| **I3** | Indoor | NLOS | Stationary | 2025-12-24 |
| **O1** | Outdoor | Mix | 30–50 km/h | 2025-12-05 |
| **O2** | Outdoor | Mix | 30–50 km/h | 2025-12-18 |

> **💡 Usage Notes**:
> * **D1** serves as the ideal baseline dataset and is highly recommended as the **Source Domain** for initial model training in cross-domain tasks.
> * **I1-I3** and **O1-O2** encompass Line-of-Sight (LOS) / Non-Line-of-Sight (NLOS) variations, temporal shifts, and Doppler frequency shifts caused by high-speed mobility. These are recommended as **Target Domains** to evaluate and benchmark model generalization capabilities.
