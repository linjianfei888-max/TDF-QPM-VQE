# TDF-QPM VQE Simulations (Public Repository)

**Reproduces all numerical results in**  
*"Twisted Definition Fields: A Unified Framework for Quantum Measurement, Nonlocality, and Spacetime Emergence"*  
Submitted to *Foundations of Physics*

---

## One-Click Reproducibility

```bash
pip install -r requirements.txt
python 2-qubit.py   # 2‑qubit: Energy → 10⁻⁵, Entropy: 1.386 → 0 → 1.386
python 4-qubit.py   # 4‑qubit: Energy → 10⁻⁵, Entropy: 2.773 → 0 → 2.773

## Output
- `data/2qubit_forward.csv`, `data/2qubit_reverse.csv`
- `data/4qubit_forward.csv`, `data/4qubit_reverse.csv`

**These files are automatically generated when you run the scripts.**  
**No pre-computed data is included in this repository.**
