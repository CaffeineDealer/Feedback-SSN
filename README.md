# Feedback-SSN

Random dot motion stimulus generation and integration with stabilized supralinear networks (SSN) with feedback connections.

## Files

- **main.py** - Main simulation script that processes directional motion stimuli through MT and MST neural populations
- **setup.py** - Random dot motion stimulus generation (adapted from Matlab)
- **ssn.py** - SSN implementation with tuned excitatory/inhibitory connectivity matrices and additional feedback input
- **polarmosaic.py** - Polar plot visualization for directional tuning curves (adapted from Matlab)

## Usage

```python
python main.py
```

Generates random dot motion stimuli at different directions, processes through MT/MST visual areas, and visualizes directional selectivity as a set of tuning wheels.

## Dependencies

- numpy
- matplotlib

## Credits

`setup.py` and `polarmosaic.py` adapted from Matlab scripts by [Shahab Bakhtiari](https://github.com/ShahabBakht/VPL-Model)