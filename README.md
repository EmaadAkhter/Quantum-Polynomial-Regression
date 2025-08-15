# Quantum Polynomial Regression
![Quantum Regression](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/quantum_regression.png)


A quantum machine learning implementation for polynomial regression using PennyLane and variational quantum circuits (VQCs).

## Overview

This project implements a quantum neural network that learns to approximate polynomial functions using quantum circuits. The model leverages quantum superposition and entanglement to potentially capture complex non-linear relationships in data.

## Features

- **Quantum Circuit Architecture**: Parameterized quantum circuit with rotation gates and entangling layers
- **Data Encoding**: Angle encoding to map classical data into quantum states
- **Variational Training**: Uses gradient-based optimization with quantum parameter shift rules
- **Automatic Scaling**: Built-in data normalization for inputs and targets
- **Performance Metrics**: MSE and R² score evaluation
- **Visualization**: Automatic plotting of results with predictions vs actual data

## Architecture

The quantum circuit consists of:

1. **Data Encoding Layer**: 
   - RY and RZ rotations proportional to input features
   - Hadamard gates for superposition

2. **Variational Layers** (repeated):
   - RX, RY, RZ rotation gates with trainable parameters
   - CNOT gates for entanglement between adjacent qubits
   - Ring connectivity for additional entanglement

3. **Measurement**: 
   - Expectation value of Pauli-Z tensor product on first two qubits

## Installation

```bash
pip install pennylane
pip install scikit-learn
pip install matplotlib
pip install numpy
```

For GPU acceleration (optional):
```bash
# For NVIDIA GPUs (requires CUDA and cuQuantum SDK)
pip install pennylane-lightning-gpu

# Note: lightning.gpu requires NVIDIA CUDA-capable GPUs (SM 7.0/Volta or newer)
# For AMD GPUs, use lightning.kokkos backend instead
# Apple Silicon Macs (M1/M2/M3/M4): No GPU acceleration currently supported
# - PennyLane lightning simulators don't yet support Apple's Metal framework
# - Use the standard lightning.qubit device on macOS
```

## Usage

### Basic Example

```python
import numpy as np
from quantum_regression import QuantumPolynomialRegression

# Generate sample data
X = np.linspace(-2, 2, 100)
y = 0.5 * X**3 + 2 * X**2 + X + 1 + np.random.normal(0, 0.5, len(X))

# Create and train model
model = QuantumPolynomialRegression(n_qbit=6, n_layers=4)
model.fit(X, y, epochs=200)

# Make predictions
predictions = model.predict_new(X)

# Plot results
model.plot_results(X, y)
```

### Custom Configuration

```python
# Create model with custom parameters
model = QuantumPolynomialRegression(
    n_qbit=8,        # Number of qubits
    n_layers=6       # Number of variational layers
)

# Train with custom epochs
model.fit(X, y, epochs=500)
```

## Parameters

### QuantumPolynomialRegression

- `n_qbit` (int, default=6): Number of qubits in the quantum circuit
- `n_layers` (int, default=4): Number of variational layers
- `epochs` (int, default=200): Number of training iterations

### Key Methods

- `fit(X, y, epochs=200)`: Train the quantum model
- `predict_new(X)`: Make predictions on new data
- `plot_results(X, y, save_path)`: Visualize predictions vs actual data

## Performance Considerations

### Quantum Advantage
- **Expressivity**: Quantum circuits can represent complex function mappings
- **Parameter Efficiency**: Exponential Hilbert space with linear parameter growth
- **Entanglement**: Captures non-local correlations in data

### Computational Requirements
- Training time scales with number of qubits and circuit depth
- Classical simulation becomes exponentially expensive for large qubit counts
- Consider using quantum hardware or GPU acceleration for larger models

## Example Output

```
Training...
Epoch 0: Loss = 0.8234
Epoch 50: Loss = 0.3456
Epoch 100: Loss = 0.2103
Epoch 150: Loss = 0.1845
MSE: 0.2847, R²: 0.8923
Plot saved to: quantum_regression.png
```

## Results & Visualizations

### 1. Quantum vs Classical Comparison
Comparison between quantum and classical polynomial regression performance:

![Quantum vs Classical Regression](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/classical_vs_quantum.png)

**Key Findings:**
- Quantum model achieves R² = 0.988 with MSE = 0.157
- Classical model achieves R² = 0.991 with MSE = 0.124
- Both models successfully capture the cubic polynomial trend
- Quantum approach shows competitive performance with potential for quantum advantage

### 2. Function Type Analysis
Performance across different polynomial functions:

![Function Analysis](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/different_functions_comparison.png)

**Results Summary:**
- **Cubic Polynomial** (R² = 0.985): Successfully captures complex cubic relationships
- **Quartic Polynomial** (R² = 0.979): Excellent fit for 4th degree polynomials
- **Sine Polynomial** (R² = 0.955): Good approximation of trigonometric functions
- **Complex Polynomial** (R² = 0.982): Handles multi-modal functions effectively
- **Exponential Decay** (R² = 0.470): Moderate performance on exponential functions

### 3. Circuit Depth Analysis
Impact of variational layers on model performance:

![Layer Analysis](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/layer_comparison.png)

**Observations:**
- **2 Layers**: R² = 0.752, MSE = 1.555 (underfitting)
- **4 Layers**: R² = 0.982, MSE = 0.115 (optimal performance)
- **6 Layers**: R² = 0.981, MSE = 0.117 (slight diminishing returns)
- **8 Layers**: R² = 0.982, MSE = 0.110 (marginal improvement)

**Recommendation**: 4-6 layers provide optimal balance between performance and computational cost.

### 4. Noise Robustness Study
Model performance under different noise conditions:

![Noise Analysis](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/noise_robustness.png)

**Noise Tolerance:**
- **σ = 0.1**: R² = 0.997 (excellent performance)
- **σ = 0.3**: R² = 0.983 (robust to moderate noise)
- **σ = 0.5**: R² = 0.938 (good resilience)
- **σ = 0.8**: R² = 0.847 (degraded but acceptable)

The quantum model demonstrates strong noise resilience, maintaining performance even with significant data corruption.

### 5. Performance Metrics Summary

![Performance Summary](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/performance_summary.png)

**Key Insights:**
- **Qubit Scaling**: Performance remains stable across 3-8 qubits
- **Layer Optimization**: 4+ layers required for optimal performance
- **Noise Robustness**: Linear degradation with increasing noise levels

### 6. Qubit Scaling Analysis
Performance across different numbers of qubits:

![Qubit Analysis](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/qubit_comparison.png)

**Scaling Results:**
- **3 Qubits**: R² = 0.977, MSE = 0.144 (minimal circuit)
- **4 Qubits**: R² = 0.980, MSE = 0.125 (good balance)
- **6 Qubits**: R² = 0.979, MSE = 0.130 (standard configuration)
- **8 Qubits**: R² = 0.983, MSE = 0.105 (maximum tested)

### 7. Training Convergence
Comparison of training curves for different configurations:

![Training Curves](https://github.com/EmaadAkhter/Quantum-Polynomial-Regression/blob/main/plots/training_curves.png)

**Convergence Analysis:**
- All configurations converge within 200 epochs
- Larger circuits show slower initial convergence but better final performance
- Training loss follows exponential decay pattern
- No evidence of overfitting in tested configurations

## Technical Details

### Data Preprocessing
- Input features normalized to [0, 1] range
- Target values scaled to [-0.9, 0.9] to match quantum measurement range
- Automatic train/test split (80/20) for evaluation

### Optimization
- Adam optimizer with learning rate 0.08
- Adjoint differentiation method for efficient gradient computation
- Parameter initialization from normal distribution (μ=0, σ=0.05)

### Circuit Design
- Angle encoding: `RY(x * π)` and `RZ(x * π/2)`
- Alternating CNOT pattern for entanglement
- Ring topology for global connectivity

## Benchmark Results

| Configuration | R² Score | MSE | Training Time | Comments |
|--------------|----------|-----|---------------|----------|
| 3Q, 4L | 0.977 | 0.144 | ~45s | Minimal viable |
| 4Q, 4L | 0.980 | 0.125 | ~60s | Recommended |
| 6Q, 4L | 0.979 | 0.130 | ~90s | Standard |
| 8Q, 5L | 0.983 | 0.105 | ~180s | High performance |

*Q = Qubits, L = Layers. Times on standard CPU.*

## Extensions

### Potential Improvements
1. **Multi-dimensional Input**: Extend encoding for multiple features
2. **Advanced Ansätze**: Implement hardware-efficient or problem-inspired circuits
3. **Regularization**: Add quantum noise or parameter penalties
4. **Ensemble Methods**: Combine multiple quantum models
5. **Real Hardware**: Deploy on quantum processors (IBM, Rigetti, IonQ)

### Research Applications
- Quantum advantage demonstrations
- Hybrid classical-quantum algorithms
- Quantum feature maps exploration
- Noise resilience studies

## Dependencies

- `pennylane`: Quantum machine learning framework
- `numpy`: Numerical computations
- `scikit-learn`: Classical ML utilities and metrics
- `matplotlib`: Plotting and visualization

## Citation

If you use this work in your research, please cite:

```bibtex
@software{quantum_polynomial_regression,
  title={Quantum Polynomial Regression with Variational Quantum Circuits},
  author={[Emaad Ansari]},
  year={2025},
  url={https://github.com/EmaadAkhter/Quantum-Polynomial-Regression}
}
```

## License

This project is open source. Please cite appropriately if used in research.

## References

1. Schuld, M., & Petruccione, F. (2018). *Supervised learning with quantum computers*
2. Biamonte, J., et al. (2017). Quantum machine learning. *Nature*, 549(7671), 195-202
3. PennyLane Documentation: https://pennylane.ai/
4. Benedetti, M., et al. (2019). Parameterized quantum circuits as machine learning models. *Quantum Science and Technology*, 4(4), 043001

## Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- Alternative quantum encodings
- Hardware deployment scripts
- Benchmarking studies

*For questions or issues, please open a GitHub issue or contact the maintainers.*
